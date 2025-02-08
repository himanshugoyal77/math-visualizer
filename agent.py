import matplotlib.pyplot as plt
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import os
import ast

def create_bar_chart(data_str: str, title: str = "Bar Chart"):
    """
    Creates a bar chart from a string representation of a dictionary and saves it as an image.
    
    Args:
        data_str (str): String representation of dictionary where keys are categories, values are numerical values.
        title (str): Title of the bar chart.

    Returns:
        str: Path to the saved image.
    """
    # Convert string representation of dictionary to actual dictionary
    try:
        data = ast.literal_eval(data_str)
    except:
        raise ValueError("Invalid data format. Please provide a valid dictionary string.")
    
    categories = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(8, 5))
    plt.bar(categories, values, color='skyblue')
    plt.title(title)
    plt.xlabel("Categories")
    plt.ylabel("Values")
    plt.xticks(rotation=45)
    plt.savefig("bar_chart.png")
    plt.close()
    
    return "bar_chart.png"

def create_pie_chart(data_str: str, title: str = "Pie Chart"):
    """
    Creates a pie chart from a string representation of a dictionary and saves it as an image.

    Args:
        data_str (str): String representation of dictionary where keys are categories, values are numerical values.
        title (str): Title of the pie chart.

    Returns:
        str: Path to the saved image.
    """
    # Convert string representation of dictionary to actual dictionary
    try:
        data = ast.literal_eval(data_str)
    except:
        raise ValueError("Invalid data format. Please provide a valid dictionary string.")
    
    categories = list(data.keys())
    values = list(data.values())

    plt.figure(figsize=(6, 6))
    plt.pie(values, labels=categories, autopct="%1.1f%%", startangle=140, colors=['lightblue', 'lightcoral', 'lightgreen'])
    plt.title(title)
    plt.savefig("pie_chart.png")
    plt.close()
    
    return "pie_chart.png"

bar_chart_tool = Tool(
    name="Bar Chart Generator",
    func=create_bar_chart,
    description="Generates a bar chart from a dictionary of categories and numerical values. Input should be a string representation of a dictionary.",
)

pie_chart_tool = Tool(
    name="Pie Chart Generator",
    func=create_pie_chart,
    description="Generates a pie chart from a dictionary of categories and numerical values. Input should be a string representation of a dictionary.",
)

llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_KEY")
)

agent = initialize_agent(
    tools=[bar_chart_tool, pie_chart_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# The data is already a string representation of a dictionary, so no need to convert it
response = agent.run("Generate a bar chart from the following data: {'A': 10, 'B': 20, 'C': 30}")
print(response)