import os
from llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('DASHSCOPE_API_KEY')

Settings.llm = OpenAILike(
    model="qwen-turbo",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    is_chat_model=True
)
# Settings.llm = OpenAI(model="gpt-4.1-nano-2025-04-14")
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v3",
    api_key=api_key
)

# 导入FAISS相关模块
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, VectorStoreIndex


# 加载数据
from llama_index.core import SimpleDirectoryReader # load documents
documents = SimpleDirectoryReader(f"/Users/caojingkang/PycharmProjects/RAG_Project/rag_app/性能调优/09混合检索/data/").load_data()

# 创建FAISS索引 (使用FlatL2替代IVF以简化测试)
d = 512  # 与text-embedding-3-small维度匹配
faiss_index = faiss.IndexFlatL2(d)  # 改用L2距离（更通用）

# 创建向量存储
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 构建索引
faiss_vector_index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, show_progress=True
)

# 创建检索器并查询
faiss_retriever = faiss_vector_index.as_retriever(similarity_top_k=2)
faiss_retrieved_nodes = faiss_retriever.retrieve("What happened at Viaweb and Interleaf?")

# 显示结果
print("FAISS检索结果:")
for node in faiss_retrieved_nodes:
    # display_source_node(node, source_length=5000)
    print(node)

# BM25与FAISS结果比较
# print("\nBM25 vs FAISS 比较:")
# print(f"BM25检索到{len(retrieved_nodes)}个节点，FAISS检索到{len(faiss_retrieved_nodes)}个节点")
#
# # 提取相似度分数进行比较
# bm25_scores = [node.score for node in retrieved_nodes]
# faiss_scores = [node.score for node in faiss_retrieved_nodes]
# print(f"BM25平均相似度: {sum(bm25_scores)/len(bm25_scores):.4f}")
# print(f"FAISS平均相似度: {sum(faiss_scores)/len(faiss_scores):.4f}")