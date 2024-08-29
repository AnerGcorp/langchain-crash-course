from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load the environment variables
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# SystemMessage:
#   - Message for priming AI behavior, usually passed in as the first of a sequence of input messages
# HumanMessage:
#   - Message from a human to the AI model

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 91 divided by 9")
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

# AI message:
#  - Message from the AI model to the user
messages = [
    SystemMessage(content="Solve the following math problems and provide the answers in markdown format with steps"),
    HumanMessage(content="What is 81 divided by 9"),
    AIMessage(content="81 divided by 9 is 9"),
    HumanMessage(content="What is 91 divided by 9")
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")