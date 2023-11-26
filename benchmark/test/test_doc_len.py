from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
import time

st = time.perf_counter_ns()
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
ed = time.perf_counter_ns()

print(f"load time: {(ed-st)/1e9}s")

docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

print(len(split_docs))