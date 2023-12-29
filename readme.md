# 先导杯（中南赛区）—**图卷积神经网络推理优化**

官网连接：[先导杯计算应用大奖赛 (sugon.com)](https://cas-pra.sugon.com/detail.html?tournament_id=33)

## 解题思路

1. 主要优化三个算子：

   a. 畸形矩阵乘（Mx128）*（128x16）

   b. 稀疏矩阵乘（CSR压缩方式）

   c. LogsoftMax

   

   矩阵是行主序存储的

### 1. 畸形矩阵乘

针对右乘矩阵128x16的大小（$ 128 \times 16 \times 8B / 1024 = 16K $），考虑把整个右乘矩阵一次性加在到Shared Memory中。

针对左乘矩阵，一个thread读取一行数据，即每个thread需要完成的计算：$1\times128$的向量和$128\times16$ 矩阵相乘。

为了实现计算和128维的向量的读取的重合，本方法采用了分块读取128维的向量。

### 2. 稀疏矩阵乘



### 3. LogsoftMax
