from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langsmith import Client
from ddgs import DDGS
import requests
import os

load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "ReAct Agent"
WEATHER_API = os.environ["WEATHER_API"]

# Tool 1: Web Search
search_tool = DuckDuckGoSearchRun()

# Tool 2: Weather Data
@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API}'
    response = requests.get(url)
    print(response.json())
    return response.json()

# LLM
llm = ChatGroq(model='llama-3.3-70b-versatile')

client = Client()
prompt = client.pull_prompt("hwchase17/react")

# Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Q1: What is the release date of Dhurandhar: The Revenge movie?
# Q2: What is the current temp of Pune?
# Q3: Identify the birthplace city of Narendra Modi, and give its current temperature.

# Invoke the agent
response = agent_executor.invoke({"input": "Identify the birthplace city of Narendra Modi, and give its current temperature"})
print(response)

print(response['output'])