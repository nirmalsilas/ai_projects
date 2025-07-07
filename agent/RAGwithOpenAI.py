import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")



doc = WebBaseLoader("https://en.wikipedia.org/wiki/Martin_Luther_King_Jr.").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(doc)

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)

llm = ChatOpenAI()



prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context provided. If the answer is not in the context, say "I don't know".
        <contex>
        Context: {context}
        </contex>

        """
)

document_chain = create_stuff_documents_chain(llm, prompt)

document_chain.invoke({
    "input": "who was martin luther king'?",
    "context": [Document(page_content="Martin Luther King Jr. was an American Baptist minister and activist who became the most visible spokesperson and leader in the American civil rights movement from 1955 until his assassination in 1968.")]
})

retriver  = vector_store.as_retriever()
retriver_chain = create_retrieval_chain(retriver, document_chain)

response = retriver_chain.invoke({"input": "who was martin luther king'?"})
print(response)

#outputparser = StrOutputParser()

#chain = prompt | llm | outputparser
#query = "who was martin luther king'?"
#response = chain.invoke({"input": query})
#print(response)



 


