from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.schema.runnable import RunnablePassthrough

from langchain.memory import ConversationBufferMemory

from langchain_core.output_parsers import StrOutputParser



llm = ollama = Ollama(model="gemma3")


embedding = OllamaEmbeddings(model="gemma3")

session_db = {}



# get chat session id
def get_session_memory(session_id:str) -> ConversationBufferMemory:

    if session_id not in session_db:
        session_db[session_id] = ConversationBufferMemory()
    return session_db[session_id]

def get_chat_runnable(session_id: str):
    memory = get_session_memory(session_id)

    prompt = ChatPromptTemplate.from_template(
        "Assistant\n"
        "Conversation history:\n{history}\n"
        "User: {input}\n"
        "Assistant:"
    )

    def get_history(_):
        return memory.load_memory_variables({})["history"]
    parser = StrOutputParser()
    runnable = (
        {
            "input": RunnablePassthrough(),
            "history": get_history
        }
        | prompt
        | llm
        | parser
    )
    return runnable, memory


def chat(session_id: str, user_input: str):
    runnable, memory = get_chat_runnable(session_id)
    response = runnable.invoke(user_input)
    memory.save_context({"input": user_input}, {"output": str(response)})
    return response



if __name__ == "__main__":

    import uuid
    print("starting your chatbot")

    default_session = str(uuid.uuid4())
    print(f"default_session: {default_session}")

    while True:
        session_input = input("\nSession ID (press Enter for default): ")
        session_id = session_input if session_input else default_session

        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat(session_id, user_input)
        print(f"Bot: {response}")







    



