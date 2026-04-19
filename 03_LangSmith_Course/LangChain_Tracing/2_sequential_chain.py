from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "Sequential LLM App"

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text: \n{text}',
    input_variables=['text']
)

model1 = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.7)

model2 = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.5
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

config = {
    'run_name': 'sequential-chain',
    'tags': ['sequential-chain', 'unemployment-report', 'summarization', 'llm-app'],
    'metadata': {
        'model1': 'llama-3.3-70b-versatile',
        'model1_temperature': 0.7,
        'model2': 'meta-llama/llama-3-8b-instruct',
        'model2_temperature': 0.5,
        'parser': 'StrOutputParser'
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)
print(result)