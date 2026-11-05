from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from langchain.agents import create_agent

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    param num: The number to be tripled.
    return: The tripled value of the input number.
    """
    return num * 3

tools = [TavilySearch(max_results=1), triple]

# This is a llm model that can be used to run the ReAct flow. Here we have control on how the llm model uses the tools. This runnable interface is exported to nodes.py so that it can be used in the StateGraph flow. The llm model is bound to the tools so that it can use them when needed.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# This is a compiled state graph that can be used to run the ReAct flow. Here we do not have control on how the llm model uses the tools.
# agent = create_agent(model="gpt-4o-mini", tools=tools)