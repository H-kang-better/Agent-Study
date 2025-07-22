import numpy as np
import faiss
import time

# 设置随机种子
np.random.seed(42)

# 构造数据集
d = 128                             # 向量维度
nb = 100000                         # 数据库大小
nq = 1000                           # 查询数量
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 构建原始 HNSW 索引
index_hnsw = faiss.IndexHNSWFlat(d, 32)
index_hnsw.hnsw.efConstruction = 200
index_hnsw.verbose = False
print("开始训练原始 HNSW...")
start_time = time.time()
index_hnsw.add(xb)
end_time = time.time()
print(f"原始 HNSW 训练耗时: {end_time - start_time:.2f}s")

# 查询测试
k = 10
D, I = index_hnsw.search(xq, k)
print("原始 HNSW top-10 查询结果（前5个查询）:")
print(I[:5])

# 获取内存占用
print(f"原始 HNSW 内存占用: {index_hnsw.ntotal * index_hnsw.d * 4 / (1024 ** 2):.2f} MB")


# 构建 HNSW + SQ8 索引
res = faiss.StandardGpuResources() if 'gpu' in faiss.__version__ else None

# 使用 HNSW 作为量化器
quantizer = faiss.IndexHNSWFlat(d, 32)
quantizer.hnsw.efConstruction = 200

# 创建带 SQ8 的索引
index_sq8 = faiss.IndexIVFPQ(quantizer, d, nb // 100, 4, 8)  # PQ 参数可调
index_sq8.nprobe = 10
index_sq8.verbose = False

# 训练索引
print("开始训练 HNSW+SQ8...")
start_time = time.time()
index_sq8.train(xb)
index_sq8.add(xb)
end_time = time.time()
print(f"HNSW+SQ8 训练耗时: {end_time - start_time:.2f}s")

# 查询测试
D2, I2 = index_sq8.search(xq, k)
print("HNSW+SQ8 top-10 查询结果（前5个查询）:")
print(I2[:5])

# 获取内存占用
print(f"HNSW+SQ8 内存占用: {index_sq8.ntotal * index_sq8.d * 1 / (1024 ** 2):.2f} MB")  # SQ8每个向量约1字节

# 回忆率计算（简单比较 Top-1 是否一致）
recall_top1 = np.mean(I[:, 0] == I2[:, 0])
print(f"Top-1 Recall between HNSW and HNSW+SQ8: {recall_top1 * 100:.2f}%")