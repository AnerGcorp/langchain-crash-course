from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load the environment variables
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")


# Invoke the model with a message
result = model.invoke("What is the capital of France?")
print("Full response: \n", result)
print("Content only: \n", result.content)