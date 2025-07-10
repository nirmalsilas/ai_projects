from langchain_community.document_loaders import TextLoader


txtloader = TextLoader("sample.txt")
document = txtloader.load()
#print(document)


#splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len
)
texts = text_splitter.split_documents(document)

print(texts[0].page_content)

#ollama embedding
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#vector store using chroma
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    texts,
    embeddings,
    collection_name="sample_collection"
)   

#qquery using ollama
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
# Run a query
query = "create a more detailed writing on similar theme"
result = qa.invoke(query)
print(result)














