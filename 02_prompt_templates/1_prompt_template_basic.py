from langchain.prompts import ChatOpenTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# PART 1: Create a ChatPromptTemplate using a template string
# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate(template)

# print("---- Prompt from Template ----")
# prompt = prompt_template.invoke({"topice": "cats"})
# print(prompt)

# # PART 2: Prompt with Multiple Placeholder
# template_multiple = """You are a helpful AI assitant.
# Human: Tell me a {adjective} story about a {animal}.
# Assistant: """
# prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
# prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
# print("\n---- Prompt with Multiple Placeholders ----")
# print(prompt)

# # PART 3: Prompt with System and Human Message (Using Tuples)
# messages = [
#     ('system', 'You are a comedian who tells jokes about {topi}')
#     ('human', 'Tell me a {joke_count} jokes')
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "cats", "joke_count": "three"})
# print(prompt)

# # Extra Information about Part 3.
# # This does work:
# message = [
#     ('system', 'You are a comedian who tells jokes about {topic}'),
#     HumanMessage(content="Tell me 3 jokes")
# ]
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "lawyers"})
# print(prompt)

# This does not work:
messages = [
    ('system', 'You are a comedian who tells jokes about {topic}'),
    HumanMessage(content="Tell me {joke_count} jokes")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": "three"})
print("\n---- Prompt with System and Human Message (Tuple) ----")
print(prompt)