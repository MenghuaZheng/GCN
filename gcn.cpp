#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <cmath>

using namespace std;
typedef std::chrono::time_point<std::chrono::steady_clock> TimePoint;

#define W_IN_DIM 128
#define W_OUT_DIM 16

#define K_TILE 16

#define BLOCK_DIM 256

#define WARP_SIZE_MAX 64 //hardware
#define WARP_SIZE 64 //logsoftmax
#define WARP_SIZE_AX 16 // AX 8

#define tf 1

#define TILE_N_AX 8


int v_num = 0;
int e_num = 0;
int F0 = 0, F1 = 0;
// coo graph
vector<vector<int>> edge_index;
vector<vector<double>> edge_val;
vector<int> degree;
vector<int> raw_graph;

// csr graph;
int *nodes_index;
int *edges;
double *edges_value;

// layer
double *X0, *W1, *X1, *X1_inter;
double *X16_1;

// layer on gpu
double *d_X0, *d_W1, *d_X1, *d_X1_inter;
double *d_X16_1;

// csr graph on gpu
int *d_index, *d_edges;
double *d_edges_val;

struct __align__(16) MD{
    double max_tmp;
    double sum_tmp;
};

void readGraph(char *fname)
{
    ifstream infile(fname);
    int source;
    int end;
    infile >> v_num >> e_num;
    while (!infile.eof())
    {
        infile >> source >> end;
        if (infile.peek() == EOF)
            break;
        raw_graph.push_back(source);
        raw_graph.push_back(end);
    }
}

void to_csr()
{

    nodes_index = (int *)malloc(v_num * sizeof(int) + 1);

    int sum = 0;
    for (int i = 0; i < v_num; i++)
    {
        nodes_index[i] = sum;
        sum += degree[i];
    }
    nodes_index[v_num] = sum;

    edges = (int *)malloc(e_num * sizeof(int));
    for (int i = 0; i < v_num; i++)
    {
        memcpy(edges + nodes_index[i], edge_index[i].data(), sizeof(int) * edge_index[i].size());
    }

    edges_value = (double *)malloc(e_num * sizeof(double));
    for (int i = 0; i < v_num; i++)
    {
        memcpy(edges_value + nodes_index[i], edge_val[i].data(), sizeof(double) * edge_val[i].size());
    }
}

void raw_graph_to_AdjacencyList()
{
    int src;
    int dst;
    edge_index.resize(v_num);
    edge_val.resize(v_num);
    degree.resize(v_num, 0);

    for (int i = 0; i < raw_graph.size() / 2; i++)
    {
        src = raw_graph[2 * i];
        dst = raw_graph[2 * i + 1];
        edge_index[dst].push_back(src);
        degree[src]++;
    }
}

void edgeNormalization()
{
    for (int i = 0; i < v_num; i++)
    {
        for (int j = 0; j < edge_index[i].size(); j++)
        {
            double val = 1 / sqrt(degree[i]) / sqrt(degree[edge_index[i][j]]);
            edge_val[i].push_back(val);
        }
    }
}

void readdouble(char *fname, double *&dst, int num)
{
    dst = (double *)malloc(num * sizeof(double));
    FILE *fp = fopen(fname, "rb");
    fread(dst, num * sizeof(double), 1, fp);
    fclose(fp);
}

void initdouble(double *&dst, int num)
{
    dst = (double *)malloc(num * sizeof(double));
    memset(dst, 0, num * sizeof(double));
}

__global__ void XW_(int in_dim, int out_dim, double *in_X, double *out_X, double *W, int v_num)
{

    int tid = threadIdx.x + blockIdx.x * blockDim.x; // 控制v_vum
    int btid = threadIdx.x;
    if (tid >= v_num)
        return;
    
    double in_tile[K_TILE];
    double out_tile[W_OUT_DIM];

#pragma unroll
    for (int i = 0; i < out_dim; ++i) {
        out_tile[i] = 0.0f;
    }

    double *tmp_in_X = in_X;
    double *tmp_out_X = out_X;
    double *tmp_W = W;

    __shared__ double smem_w[W_IN_DIM*W_OUT_DIM];
// global -> smem
#pragma unroll
    for(int i = 0; i < W_IN_DIM*W_OUT_DIM/blockDim.x; i++){
        smem_w[btid + i * blockDim.x] = tmp_W[btid + i * blockDim.x];
    }
    __syncthreads();

    for(int i = 0; i < in_dim/K_TILE; i++){
#pragma unroll        
        for(int j = 0; j < K_TILE; j++){
            in_tile[j] = tmp_in_X[tid * in_dim + i*K_TILE + j];
        }

        for (int j = 0; j < out_dim; j++){
#pragma unroll
            for (int k = 0; k < K_TILE; k++)
            {
                out_tile[j] += in_tile[k] * smem_w[(i*K_TILE+ k) * out_dim + j];
            }
        }

    }

#pragma unroll
    for(int i = 0; i < out_dim; i++){
        tmp_out_X[tid * out_dim + i] = out_tile[i];
    }
}

__global__ void AX_(int dim, double *in_X, double *out_X, int *index, int *edges, double *edges_val, int v_num){
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int wtid = threadIdx.x % WARP_SIZE_AX;
    int wid = threadIdx.x / WARP_SIZE_AX;
    int by = blockIdx.y;
    int r = gtid / WARP_SIZE_AX;
    
    if (r >= v_num) return;

    double out[TILE_N_AX];

#pragma unroll
    for(int i = 0; i < TILE_N_AX; i++){
        out[i] = 0.0f;
    }

    int *nbrs = &edges[index[r]];
    double *nbrs_val = &edges_val[index[r]];
    int degree = index[r + 1] - index[r];

    int nbr;
    double reg_nbrs_val; 
    for (int j = 0; j + wtid < degree; j += WARP_SIZE_AX){
        nbr = nbrs[j + wtid];
        reg_nbrs_val = nbrs_val[j + wtid];
#pragma unroll 
        for (int k = 0; k < TILE_N_AX; k++){
            out[k] += in_X[nbr * dim + by*TILE_N_AX + k] * reg_nbrs_val;
        }
    }

#pragma unroll 
    for(int i = WARP_SIZE_AX / 2; i > 0; i /= 2){
#pragma unroll 
        for(int k = 0; k < TILE_N_AX; k++){
            out[k] += __shfl_down(out[k], i, WARP_SIZE_MAX);
        }
    }

    if(wtid == 0){
#pragma unroll
        for(int k = 0; k < TILE_N_AX; k++){
            out_X[dim * r + by * TILE_N_AX + k] = out[k];
        }
    }
}

__device__ __forceinline__ MD reduce_md_op(MD a, MD b){
    bool a_bigger = (a.max_tmp > b.max_tmp);
    MD bigger = a_bigger ? a : b;
    MD smaller = a_bigger ? b : a;
    MD res;
    res.sum_tmp = bigger.sum_tmp + smaller.sum_tmp * exp(smaller.max_tmp - bigger.max_tmp);
    res.max_tmp = bigger.max_tmp;

    return res;
}

__global__ void logsoftMax(int v_num, int dim, double *X, double *X_OUT){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    int wave_id = tid / 8;
    int wave_tid = tid % 8;
    
    if(wave_id >= v_num) return;

    int tx = threadIdx.x % (WARP_SIZE_MAX / (W_OUT_DIM/2));
    int ty = threadIdx.x / (WARP_SIZE_MAX / (W_OUT_DIM/2));

    double data1 = X[blockIdx.x * blockDim.x * 2 + ty* W_OUT_DIM + tx];
    double data2 = X[blockIdx.x * blockDim.x * 2 + ty* W_OUT_DIM + tx + 8];

    MD md_input;
    MD md_partial;
    md_input.max_tmp = data1;
    md_input.sum_tmp = 1.0f;

    md_partial.max_tmp = data2;
    md_partial.sum_tmp = 1.0f;
    
    double sum = data1 + data2;
    md_partial = reduce_md_op(md_input, md_partial);    

#pragma unroll
    for(int i = 1; i < W_OUT_DIM/2; i *= 2){
        md_input.max_tmp = __shfl_down(md_partial.max_tmp, i, 64);
        md_input.sum_tmp = __shfl_down(md_partial.sum_tmp, i, 64);
        sum += __shfl_down(sum, i, 64);

        md_partial = reduce_md_op(md_input, md_partial);
    }

    if(wave_tid == 0) X_OUT[wave_id] = sum - W_OUT_DIM*md_partial.max_tmp - W_OUT_DIM * log(md_partial.sum_tmp);
}

void LogSoftmax(int dim, double *X)
{

    for (int i = 0; i < v_num; i++)
    {
        double max = X[i * dim + 0];
        for (int j = 1; j < dim; j++)
        {
            if (X[i * dim + j] > max)
                max = X[i * dim + j];
        }

        double sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += exp(X[i * dim + j] - max);
        }
        sum = log(sum);

        for (int j = 0; j < dim; j++)
        {
            X[i * dim + j] = X[i * dim + j] - max - sum;
        }
    }
}

double MaxRowSum(double *X, int dim)
{
    double max = -__FLT_MAX__;

    for (int i = 0; i < v_num; i++)
    {
        double sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += X[i * dim + j];
        }
        if (sum > max){
            max = sum;
        }
            
    }
    return max;
}

void freedoubles()
{
    free(X0);
    free(W1);
    free(X1);
    free(X1_inter);
    free(nodes_index);
    free(edges);
    free(edges_value);
    hipFree(d_X0);
    hipFree(d_X1_inter);
    hipFree(d_W1);
    hipFree(d_X1);

    hipFree(d_index);
    hipFree(d_edges);
    hipFree(d_edges_val);
}


void initGPUMemory()
{

    hipFree(0);

    hipMalloc(&d_X0, v_num * F0 * sizeof(double));
    hipMemcpy(d_X0, X0, v_num * F0 * sizeof(double), hipMemcpyHostToDevice);

    hipMalloc(&d_X1_inter, v_num * F1 * sizeof(double));
    hipMemcpy(d_X1_inter, X1_inter, v_num * F1 * sizeof(double), hipMemcpyHostToDevice);

    hipMalloc(&d_W1, F0 * F1 * sizeof(double));
    hipMemcpy(d_W1, W1, F0 * F1 * sizeof(double), hipMemcpyHostToDevice);

    hipMalloc(&d_X1, F1 * v_num * sizeof(double));
    hipMemcpy(d_X1, X1, F1 * v_num * sizeof(double), hipMemcpyHostToDevice);

    //    d_index, d_edge, d_edge_val

    hipMalloc(&d_index, (v_num + 1) * sizeof(int));
    hipMemcpy(d_index, nodes_index, (v_num + 1) * sizeof(int), hipMemcpyHostToDevice);

    hipMalloc(&d_edges, e_num * sizeof(int));
    hipMemcpy(d_edges, edges, e_num * sizeof(int), hipMemcpyHostToDevice);

    hipMalloc(&d_edges_val, e_num * sizeof(double));
    hipMemcpy(d_edges_val, edges_value, e_num * sizeof(double), hipMemcpyHostToDevice);
}


double GCN()
{
    hipMemset(d_X1_inter, 0, v_num * F1 * sizeof(double));
    hipMemset(d_X1, 0, F1 * v_num * sizeof(double));
    hipMalloc(&d_X16_1, v_num * sizeof(double));
    hipHostMalloc(&X16_1, v_num * sizeof(double));
    // X16_1 = (double*)malloc(v_num * sizeof(double));

    TimePoint start = chrono::steady_clock::now();

    const int block_size = BLOCK_DIM;
    const int grid_size = (v_num / block_size / tf + 1);
    XW_<<<grid_size, block_size>>>(F0, F1, d_X0, d_X1_inter, d_W1, v_num);

    const int grid_size2 = (v_num / block_size + 1) * WARP_SIZE_AX;
    dim3 Grid2D(grid_size2, W_OUT_DIM / TILE_N_AX);
    AX_<<<Grid2D, block_size>>>(F1, d_X1_inter, d_X1, d_index, d_edges, d_edges_val, v_num); 
    
    const int grid_size1 = (v_num / block_size + 1) * 8;
    logsoftMax<<<grid_size1, block_size>>>(v_num, F1, d_X1, d_X16_1);

    hipMemcpy(X16_1, d_X16_1, sizeof(double) * v_num, hipMemcpyDeviceToHost);
 
    for(int i = 0; i < v_num; i++){
        *(X1+i*W_OUT_DIM) = *(X16_1+i);
    }

    TimePoint end = chrono::steady_clock::now();
    chrono::duration<double> l_durationSec = end - start;
    double l_timeMs = l_durationSec.count() * 1e3;

    return l_timeMs;
}

void XW_verify(int in_dim, int out_dim, double *in_X, double *out_X, double *W)
{
    double *tmp_in_X = in_X;
    double *tmp_out_X = out_X;
    double *tmp_W = W;

    for (int i = 0; i < v_num; i++)
    {   
        for (int j = 0; j < out_dim; j++)
        {
            for (int k = 0; k < in_dim; k++)
            {
                tmp_out_X[i * out_dim + j] += tmp_in_X[i * in_dim + k] * tmp_W[k * out_dim + j];
            }
        }
    }
}

void AX_verify(int dim, double *in_X, double *out_X)
{
    for (int i = 0; i < v_num; i++)

    {
        int *nbrs = &edges[nodes_index[i]];
        double *nbrs_val = &edges_value[nodes_index[i]];
        int degree = nodes_index[i + 1] - nodes_index[i];
        
        for (int j = 0; j < degree; j++)
        {
            int nbr = nbrs[j];
            for (int k = 0; k < dim; k++)
            {
                out_X[dim * i + k] += in_X[nbr * dim + k] * nbrs_val[j];
            }
        }
    }
}

void LogSoftmax_verify(int dim, double *X)
{

    for (int i = 0; i < v_num; i++)
    {
        double max = X[i * dim + 0];
        for (int j = 1; j < dim; j++)
        {
            if (X[i * dim + j] > max)
                max = X[i * dim + j];
        }

        double sum = 0;
        for (int j = 0; j < dim; j++)
        {
            sum += exp(X[i * dim + j] - max);
        }
        sum = log(sum);

        for (int j = 0; j < dim; j++)
        {
            X[i * dim + j] = X[i * dim + j] - max - sum;
        }
    }
}

bool verify(double max_sum)
{

    memset(X1_inter, 0, v_num * F1 * sizeof(double));
    memset(X1, 0, F1 * v_num * sizeof(double));

    XW_verify(F0, F1, X0, X1_inter, W1);

    // printf("Layer1 AX\n");
    AX_verify(F1, X1_inter, X1);

    // printf("Layer1 ReLU\n");
    LogSoftmax_verify(F1, X1);
    double verify_max_sum = MaxRowSum(X1, F1);
    printf("CPU_max_sum,  %6f\n", verify_max_sum);
    printf("GPU_max_sum,  %6f\n", max_sum);
    return fabs(max_sum - verify_max_sum) < 0.000001;
}


int main(int argc, char **argv)
{
    // !!! Attention !!!
    // Datasets: web-stanford ak_2010 dblp
    // Downloaded from：

    // 编译：
	//      hipify-perl gcn.cu > gcn.cpp
	//      hipcc gcn.cpp -o gcn
    //
    // 执行：仅供测试参考，队伍提交直接执行slurm.sh 即可
    //      可执行程序需接收5个参数，分别为：
	//      输入顶点特征长度F0，第一层顶点特征长度F1，图结构文件名，输入顶点特征矩阵文件名，第一层权重矩阵文件名
    //      ./gcn 128 16 graph/web-stanford_nodes_281903_edges_1992636_core_71.txt embedding/web-stanford_F0_128.bin weight/web-stanford_F0_128_F1_16.bin
    //      ./gcn 128 16 graph/com-dblp_nodes_317080_edges_1049866_core_113.txt embedding/dblp_F0_128.bin weight/dblp_F0_128_F1_16.bin
    //      ./gcn 128 16 graph/ak_2010.txt embedding/ak_2010_F0_128.bin weight/ak_2010_F0_128_F1_16.bin
    
    // 要求： 
    //      只允许修改GCN()函数里包含的代码；其余代码不允许修改，一旦发现取消成绩。

    // 评分：
    //      计算耗时显示 程序运行后会循环计算五次，评分是主要查看平均耗时。

    // 提交：
    //      查看slurm.sh 文件
    F0 = atoi(argv[1]);
    F1 = atoi(argv[2]);
    readGraph(argv[3]);
    readdouble(argv[4], X0, v_num * F0);
    readdouble(argv[5], W1, F0 * F1);
    initdouble(X1, v_num * F1);
    initdouble(X1_inter, v_num * F1);

    raw_graph_to_AdjacencyList();
    edgeNormalization();
    to_csr();
    initGPUMemory();

    double max_sum = 0, ave_timeMs = 0;
    int ROUNDs = 20;
    
    GCN();

    for (int i = 0; i < ROUNDs; i++)
    {
        // ################
        //
        ave_timeMs += GCN();
        // ################
        // Time point at the end of the computation
        // Compute the max row sum for result verification
        max_sum = MaxRowSum(X1, F1);

        // The max row sum and the computing time should be print
    }


    printf("verify\n");

    if (verify(max_sum))
    {
        printf("True\n");
    }
    else
    {
        printf("False\n");
    }

    printf("%f\n", ave_timeMs / ROUNDs);

    freedoubles();
}
