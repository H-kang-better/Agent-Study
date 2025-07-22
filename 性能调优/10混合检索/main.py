# 加载文档
from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader("./10/docs/").load_data()
print("Example document:\n", documents[0])


from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, VectorStoreIndex

URI = "http://localhost:19530"  # Milvus URI

vector_store = MilvusVectorStore(
    uri=URI,
    # token=TOKEN,
    dim=1536,  # vector dimension depends on the embedding model
    enable_sparse=True,  # enable the default full-text search using BM25
    overwrite=True,  # drop the collection if it already exists
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

import textwrap

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid", similarity_top_k=10
)
response = query_engine.query("孙悟空名字的由来？")

print(textwrap.fill(str(response), 100))

for idx, node in enumerate(response.source_nodes, 1):
        print(f"结果 {idx}: ")
        print(textwrap.fill(str(node.node.text), 100))
        print("\n")