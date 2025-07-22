from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# 1. 准备示例文档
# 我们创建一些包含专业术语的示例文本
doc_text = """
卷积神经网络（CNN）是深度学习中的一种关键模型，尤其在图像识别领域表现出色。
它的核心在于通过卷积层和池化层自动提取图像的局部特征。

与CNN不同，Transformer模型最初应用于自然语言处理（NLP）任务，
例如机器翻译。如今，它也被成功应用于计算机视觉，称为Vision Transformer。

大型语言模型（LLM）是当前AI研究的热点，它基于Transformer架构，
能够理解和生成类似人类的文本，展现出强大的推理能力。
"""
with open("../sample_tech_doc.txt", "w", encoding="utf-8") as f:
    f.write(doc_text)

# 2. 加载和切分文档
loader = TextLoader("../sample_tech_doc.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# 3. 创建向量数据库
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# --- MultiQueryRetriever 实现 ---

# 4. 初始化LLM和检索器
llm = ChatOpenAI(temperature=0, model="gpt-4o")
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# 5. 执行查询
query = "CNN是什么？"
retrieved_docs = retriever_from_llm.invoke(query)

# --- 效果分析 ---
print(f"原始查询: {query}")
print("\n--- MultiQueryRetriever 生成的查询变体 ---")
# MultiQueryRetriever 内部有日志记录生成的查询，这里我们手动展示其可能生成的查询
# 实际使用中可以通过设置 logging.basicConfig(level=logging.INFO) 查看
generated_queries = [
    "卷积神经网络的定义是什么？",
    "CNN模型在深度学习中的作用是什么？",
    "介绍一下卷积神经网络（CNN）。"
]
for i, q in enumerate(generated_queries):
    print(f"查询变体 {i+1}: {q}")

print("\n--- 最终检索到的文档内容 ---")
for doc in retrieved_docs:
    print(doc.page_content)