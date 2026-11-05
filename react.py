from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

#from langchain.agents import create_agent

load_dotenv()

@tool
def triple(num: float) -> float:
    """
    param num: The number to be tripled.
    return: The tripled value of the input number.
    """
    return num * 3

tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
#agent = create_agent(model="gpt-4o-mini", tools=tools)