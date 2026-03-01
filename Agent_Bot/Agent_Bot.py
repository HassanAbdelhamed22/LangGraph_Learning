from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

print("ENV PATH:", ENV_PATH)
print("ENV EXISTS:", os.path.exists(ENV_PATH))

load_dotenv(ENV_PATH)

print("GEMINI_API_KEY loaded?", bool(os.getenv("GEMINI_API_KEY")))
print("Key prefix:", (os.getenv("GEMINI_API_KEY") or "")[:6])


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state


graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("\nYou: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("\nYou: ")