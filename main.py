from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import HumanMessage,AIMessage
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7
)

store = {}

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Asnwer all questions to the best of your ability"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = prompt | model
config ={"configurable": {"session_id":"666"}}
with_message_history = RunnableWithMessageHistory(chain,get_session_history)

if __name__ == "__main__":
    while True:
        user_input = input(">")
        response = with_message_history.invoke(
            [
                HumanMessage(content=user_input)
            ],
            config=config
        )
        print(response.content)
