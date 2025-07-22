import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_term_vector_index(term_glossary: dict, model: SentenceTransformer) -> tuple:
    """
    将术语词库中的所有术语及其别名转换为向量，并构建FAISS索引。

    Args:
        term_glossary (dict): 结构化的术语词库。键为标准术语，值为包含'synonyms'列表的字典。
        model (SentenceTransformer): 已加载的SentenceTransformer模型实例。

    Returns:
        tuple: (faiss.Index, list) 返回构建好的FAISS索引对象和与之对应的术语列表。
    """
    terms_to_index = []
    # 遍历术语映射字典，收集所有标准术语和别名
    # 修正点：将 term_mapping_dict 修改为 term_glossary
    for standard_term, info in term_glossary.items():
        terms_to_index.append(standard_term)
        if "synonyms" in info and isinstance(info["synonyms"], list):
            terms_to_index.extend(info["synonyms"])

    unique_terms_to_index = sorted(list(set(terms_to_index)))

    print("正在生成术语向量...")
    embeddings = model.encode(unique_terms_to_index, show_progress_bar=True)

    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS索引构建完成。包含 {index.ntotal} 个向量，维度为 {dimension}。")
    return index, unique_terms_to_index

def load_embedding_model():
    """
    加载bge-small-zh-v1.5模型
    :return: 返回加载的bge-small-zh-v1.5模型
    """
    print(f"加载Embedding模型中")
    # SentenceTransformer读取绝对路径下的bge-small-zh-v1.5模型，非下载
    embedding_model = SentenceTransformer(os.path.abspath('../bge-small-zh-v1.5'))
    print(f"bge-small-zh-v1.5模型最大输入长度: {embedding_model.max_seq_length}")
    return embedding_model

# 1. 加载模型
model = load_embedding_model()

# 2. 准备术语数据
term_mapping_example = {
    "卷积神经网络": {"synonyms": ["CNN", "ConvNet"]},
    "Transformer": {"synonyms": ["transformer", "TRANSFORMER"]},
    "图像识别": {"synonyms": ["图像分类", "视觉识别"]}
}

# 3. 使用修正后的函数进行调用
faiss_index, indexed_term_list = build_term_vector_index(term_mapping_example, model)

# 4. 验证结果
print("\n--- 索引构建成功 ---")
print(f"FAISS 索引中的向量数量: {faiss_index.ntotal}")
print(f"被索引的术语列表: {indexed_term_list}")


# --- 第2部分：定义我们的核心检索函数 ---

def search_similar_terms(query_text: str, model: SentenceTransformer, index: faiss.Index, term_list: list, k: int = 5):
    """
    在FAISS索引中检索与查询文本最相似的k个术语。

    Args:
        query_text (str): 用户输入的查询词。
        model (SentenceTransformer): 用于编码查询词的模型。
        index (faiss.Index): FAISS索引对象。
        term_list (list): 与索引向量顺序一致的术语列表。
        k (int): 希望返回的最相似结果的数量。
    """
    print(f"\n--- 正在执行检索 ---")
    print(f"查询: '{query_text}'")

    # 1. 将查询文本编码为向量
    query_vector = model.encode([query_text])
    query_vector = query_vector.astype('float32')

    # 2. 在FAISS索引中执行搜索
    # index.search返回两个数组：D (distances) 和 I (indices)
    distances, indices = index.search(query_vector, k)

    # 3. 解析并打印结果
    print("检索结果:")
    for i in range(k):
        idx = indices[0][i]
        dist = distances[0][i]
        term = term_list[idx]

        # 对于IndexFlatL2，距离是平方欧氏距离，距离越小代表越相似
        print(f"  Top {i + 1}: 术语='{term}',  距离={dist:.4f} (值越小越相似)")


# 4. === 演示效果 ===

# **案例一：用别名查询标准术语**
# 目标：测试系统能否理解 "CNN" 就是 "卷积神经网络"。
search_similar_terms(query_text="CNN", model=model, index=faiss_index, term_list=indexed_term_list, k=3)

# **案例二：语义相近查询（核心优势展示）**
# 目标：查询一个不在我们词库中，但意思相近的词 "计算机视觉"。
# 预期：系统应该能找到 "图像识别" 或 "视觉识别" 等相关术语。
search_similar_terms(query_text="计算机视觉", model=model, index=faiss_index, term_list=indexed_term_list, k=3)

# **案例三：用一个更宽泛的词查询**
# 目标：查询 "语言模型"，看是否能找到更具体的 "大型语言模型" 或 "自然语言处理"。
search_similar_terms(query_text="语言模型", model=model, index=faiss_index, term_list=indexed_term_list, k=3)

# **案例四：测试对轻微噪声的容忍度**
# 目标：查询一个不存在的、略有差异的词 "变换器模型"，看是否能正确匹配到 "Transformer模型"。
search_similar_terms(query_text="变换器模型", model=model, index=faiss_index, term_list=indexed_term_list, k=3)