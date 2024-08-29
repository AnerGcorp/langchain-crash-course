from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

# Load environment variable
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are an expert product reviewer.'),
        ('human', 'List the product features of the product {product_name}')
    ]
)

# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a product expert product reviewer.'),
            ('human', 'Given these features: {features}, list the pros of these features')
        ]
    )
    return pros_template.format_prompt(features=features)

# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ('system', 'You are a product expert product reviewer.'),
            ('human', 'Given these features: {features}, list the cons of these features')
        ]
    )
    return cons_template.format_prompt(features=features)

# Combine the pros and cons analysis into final review
def combine_reviews(pros, cons):
    return f"Pros: {pros}\nCons: {cons}"

# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create the combined chain using LCEL
chain = (
    prompt_template 
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_reviews(x["branches"]["pros"], x["branches"]["cons"])) 
)

# Run the chain
result = chain.invoke({"product_name": "iPhone 13 Pro Max"})

# Output
print(result)