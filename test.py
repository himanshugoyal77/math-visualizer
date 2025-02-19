from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os

class TeachingAgent:
    def __init__(self, topic="order words"):
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_KEY")
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.topic = topic
        
        # Initial explanation prompt
        self.explain_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Provide a simple and clear explanation of {topic}. 
            Keep it concise and beginner-friendly. 
            Don't provide examples yet.
            """
        )
        
        # Example generation prompt
        self.example_prompt = PromptTemplate(
            input_variables=["topic", "chat_history"],
            template="""
            Based on the previous explanation of {topic}, provide 2-3 practical examples.
            Make the examples relatable to everyday situations.
            Previous conversation: {chat_history}
            """
        )
        
        # Quiz generation prompt
        self.quiz_prompt = PromptTemplate(
            input_variables=["topic", "chat_history"],
            template="""
            Based on the explanation and examples of {topic}, create a simple quiz question.
            Make it multiple choice (a, b, c).
            Make sure it tests understanding, not just memorization.
            Previous conversation: {chat_history}
            """
        )
        
        # Initialize chains
        self.explain_chain = LLMChain(
            llm=self.llm,
            prompt=self.explain_prompt,
            memory=self.memory
        )
        
        self.example_chain = LLMChain(
            llm=self.llm,
            prompt=self.example_prompt,
            memory=self.memory
        )
        
        self.quiz_chain = LLMChain(
            llm=self.llm,
            prompt=self.quiz_prompt,
            memory=self.memory
        )

    def start_lesson(self):
        while True:
            command = input("Enter 'start' to begin the lesson or 'quit' to exit: ").lower()
            
            if command == 'quit':
                print("Goodbye!")
                break
                
            elif command == 'start':
                # Step 1: Initial Explanation
                explanation = self.explain_chain.run(topic=self.topic)
                print("\nLet me explain", self.topic)
                print(explanation)
                
                # Step 2: Check Understanding
                while True:
                    understanding = input("\nDo you understand this explanation? (yes/no): ").lower()
                    
                    if understanding == 'no':
                        print("\nLet me explain it again in a different way...")
                        explanation = self.explain_chain.run(topic=self.topic)
                        print(explanation)
                    elif understanding == 'yes':
                        break
                    else:
                        print("Please answer with 'yes' or 'no'")
                
                # Step 3: Provide Examples
                print("\nGreat! Let me share some examples...")
                examples = self.example_chain.run(topic=self.topic)
                print(examples)
                
                # Step 4: Quiz
                print("\nNow, let's test your understanding with a quiz question:")
                quiz = self.quiz_chain.run(topic=self.topic)
                print(quiz)
                
                answer = input("\nYour answer (a/b/c): ").lower()
                
                # Note: In a real implementation, you'd want to have the LLM 
                # generate the correct answer and verify against it
                print("\nThank you for your answer! Would you like to:")
                print("1. Start over")
                print("2. Quit")
                
                choice = input("Enter your choice (1 or 2): ")
                if choice == '2':
                    print("Goodbye!")
                    break

# Create and run the teaching agent
agent = TeachingAgent(topic="order words")
agent.start_lesson()