from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import HumanMessage,AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7
)

if __name__ == "__main__":
    messages = [
        HumanMessage(content="Hello, my name is Berkay"),
        AIMessage(content="Hello Berkay, how can i help you today?"),
        HumanMessage(content="What is my name?"),
    ]
    response = model.invoke(messages)
    print(response.content)

