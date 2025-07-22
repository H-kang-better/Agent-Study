from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

# prepare the prompt template for document generation
prompt_template = """回答问题。
问题：{question}
回答："""
llm = ChatOpenAI()
# multi_llm = ChatOpenAI(n=4)
prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

# initialize the hypothetical document embedder
base_embeddings = OpenAIEmbeddings()
embeddings = HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=base_embeddings)

result = embeddings.embed_query("塞尔达传说的主角是谁？")
len(result)