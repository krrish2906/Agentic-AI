import os
import warnings

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import sqlite3

# -------------- Environment variables --------------
load_dotenv()
os.environ["LANGSMITH_PROJECT"] = "chatbot-tools"
STOCK_API_KEY = os.getenv("STOCK_API_KEY")


# ------------------ Chat State -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ------------------ Tools ------------------
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_number: float, second_number: float, operation: str) -> dict:
    """
    Simple calculator tool that performs basic arithmetic operations on two numbers
    
    Args:
        first_number: First number
        second_number: Second number
        operation: Operation to perform (add, subtract, multiply, divide)
    
    Returns:
        Dictionary with result of the operation
    """
    
    try:
        if operation == "add":
            result = first_number + second_number
        elif operation == "subtract":
            result = first_number - second_number
        elif operation == "multiply":
            result = first_number * second_number
        elif operation == "divide":
            if second_number == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_number / second_number
        else:
            return {"error": f"Unsupported operation: {operation}"}
    except Exception as e:
        return {"error": str(e)}
    return {
        "first_number": first_number, 
        "second_number": second_number, 
        "operation": operation, 
        "result": result
    }

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
    response = requests.get(url)
    return response.json()


# ------------------ LLM ------------------
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm = ChatOpenAI(
    model="openai/gpt-oss-120b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0
)
tools = [search_tool, calculator, get_stock_price]
llm_with_tools = llm.bind_tools(tools)


# ------------------ Chat Node & Tool Node ------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ------------------ Database Setup ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "chatbot.db")

# Connect to SQLite database
conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# ------------------ Graph Setup ------------------
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Compile the graph
chatbot = graph.compile(checkpointer=checkpointer)


# ------------------ Helper Functions ------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)