from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
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
    messages_ai: List[AIMessage]


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")