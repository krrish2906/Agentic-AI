from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model="llama-3.3-70b-versatile")

def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(ChatState)
checkpointer = MemorySaver()

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)