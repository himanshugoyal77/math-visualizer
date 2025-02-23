import streamlit as st
import os
import matplotlib.pyplot as plt
from io import StringIO
import contextlib
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

def execute_visualization_code(code):
    """Execute visualization code and return the plot"""
    output_buffer = StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code)
        fig = plt.gcf()
        plt.close()
        return fig, None
    except Exception as e:
        return None, str(e)

class TeachingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model='google/gemini-2.0-flash-001',
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_KEY")
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        self.explain_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Provide a simple and clear explanation of {topic}. 
            If a visualization would enhance understanding:
            1. Describe the visualization.
            2. Generate Python code using matplotlib to create it.
            3. Ensure the code uses 'import matplotlib.pyplot as plt' and 'plt.figure(figsize=(10, 6))'.
            4. Do NOT include plt.show().
            """
        )
        
        self.explain_chain = LLMChain(
            llm=self.llm,
            prompt=self.explain_prompt,
            memory=self.memory
        )
    
    def extract_visualization(self, response):
        """Extract visualization description and code from response"""
        if '```python' in response:
            parts = response.split('```python')
            desc_part = parts[0].strip()
            code_part = parts[1].split('```')[0].strip()
            return desc_part, code_part
        return response, None
    
    def explain_topic(self, topic):
        response = self.explain_chain.run(topic=topic)
        return self.extract_visualization(response)

agent = TeachingAgent()

st.title("Interactive Teaching Agent")
topic = st.text_input("Enter a topic to learn about:")
if topic:
    explanation, vis_code = agent.explain_topic(topic)
    st.subheader("Explanation")
    st.write(explanation)
    
    if vis_code:
        st.subheader("Generated Visualization Code")
        st.code(vis_code, language='python')
        fig, error = execute_visualization_code(vis_code)
        if fig:
            st.pyplot(fig)
        elif error:
            st.error(f"Error executing visualization code: {error}")
