import os
from langchain.retrievers import MergerRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from rag_app import config

# --- 1. 准备工作：设置API Key并准备数据 ---

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# 准备示例文档，我们稍微丰富一下内容以便于对比
doc_text = """
第一部分：关于卷积网络。
卷积神经网络（CNN）是深度学习中的一种关键模型，尤其在图像识别领域表现出色。
它的核心在于通过卷积层和池化层自动提取图像的局部特征。CNN的这个特性让它非常高效。

第二部分：关于Transformer。
与CNN不同，Transformer模型最初应用于自然语言处理（NLP）任务，
例如机器翻译。如今，一种被称为Vision Transformer（ViT）的变体也被成功应用于计算机视觉领域。

第三部分：关于大模型。
大型语言模型（LLM）是当前AI研究的热点，它通常基于Transformer架构，
能够理解和生成类似人类的文本，展现出强大的推理能力。
"""
with open("../../hybrid_search_doc.txt", "w", encoding="utf-8") as f:
    f.write(doc_text)

# 加载和切分文档
loader = TextLoader("../../hybrid_search_doc.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

print(f"文档已被切分为 {len(docs)} 个块。")


# --- 2. 构建两个不同的检索器 ---

# **检索器一：FAISS 向量检索器 (用于语义匹配)**
print("\n正在构建FAISS向量检索器...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("FAISS检索器构建完成。")


# **检索器二：BM25 关键词检索器 (用于精确匹配)**
print("\n正在构建BM25关键词检索器...")
# BM25Retriever可以直接从文档列表初始化，它不需要嵌入模型
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3
print("BM25检索器构建完成。")


# --- 3. 使用 MergerRetriever 合并 ---

print("\n正在初始化 MergerRetriever...")
# 创建一个检索器列表
retriever_list = [bm25_retriever, faiss_retriever]

# 初始化 MergerRetriever
# 它会自动处理并行检索和结果去重
merged_retriever = MergerRetriever(retrievers=retriever_list)
print("MergerRetriever 初始化完成。")


# --- 4. 执行查询并对比效果 ---

query = "ViT的技术细节"
print(f"\n\n--- 正在执行混合检索 ---")
print(f"查询: '{query}'")


# **为了对比，我们先看看每个检索器单独工作的结果**
print("\n--- 单独检索结果对比 ---")
bm25_results = bm25_retriever.invoke(query)
print(f"【BM25 关键词检索结果】(共 {len(bm25_results)} 条):")
for doc in bm25_results:
    print(f"  - {doc.page_content[:50]}...")

faiss_results = faiss_retriever.invoke(query)
print(f"\n【FAISS 向量检索结果】(共 {len(faiss_results)} 条):")
for doc in faiss_results:
    print(f"  - {doc.page_content[:50]}...")


# **现在看看 MergerRetriever 的混合结果**
print("\n--- MergerRetriever 混合检索结果 ---")
merged_results = merged_retriever.invoke(query)
print(f"【最终混合结果】(共 {len(merged_results)} 条，已去重):")
for doc in merged_results:
    print(f"  - {doc.page_content[:50]}...")