from langchain.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain



from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="gemma3")

session_db = {}

#get chat session id 
def get_session_id(sessionId :str) -> str:
    
    if sessionId not in session_db:
        session_db[sessionId] = ConversationBufferMemory()
    return session_db[sessionId]

def get_conversation_chain(sessionId: str):
    memory = get_session_id(sessionId)

    chain = ConversationChain(
        llm=llm,
        memory=memory 
    )
    return chain


def chat(sessionId: str, user_input: str):
    chain = get_conversation_chain(sessionId)
    response = chain.invoke({"input": user_input})
    parser = StrOutputParser()
    response = parser.parse(response)
    return response



if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        session_id = input("Enter session ID: ")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat(session_id, user_input)
        print("Bot:", response)
    



