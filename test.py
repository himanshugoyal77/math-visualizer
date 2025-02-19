models = [
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat"
]
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os

class TeachingAgent:
    def __init__(self, topic="order words"):
        self.llm = ChatOpenAI(
            model="google/gemini-2.0-flash-001",
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
            Make sure to provide different examples than before if any were given.
            Previous conversation: {chat_history}
            """
        )
        
        # Quiz generation prompt
        self.quiz_prompt = PromptTemplate(
            input_variables=["topic", "chat_history"],
            template="""
            Create a multiple choice question to test understanding of {topic}.
            Include three options (a, b, c) where only one is correct.
            Structure your response in the following format:
            QUESTION: (the question text)
            OPTIONS:
            a) (first option)
            b) (second option)
            c) (third option)
            CORRECT_ANSWER: (just the letter of the correct answer)
            EXPLANATION: (why this is the correct answer)
            HINT: (a helpful hint that doesn't give away the answer)

            Make sure the incorrect options are plausible but clearly wrong when thinking carefully.
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

    def parse_quiz_response(self, response):
        """Parse the quiz response from LLM into components"""
        lines = response.split('\n')
        quiz_data = {
            'question': '',
            'options': {},
            'correct_answer': '',
            'explanation': '',
            'hint': ''
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('QUESTION:'):
                current_section = 'question'
                quiz_data['question'] = line.replace('QUESTION:', '').strip()
            elif line.startswith('OPTIONS:'):
                current_section = 'options'
            elif line.startswith('CORRECT_ANSWER:'):
                quiz_data['correct_answer'] = line.replace('CORRECT_ANSWER:', '').strip().lower()
            elif line.startswith('EXPLANATION:'):
                current_section = 'explanation'
                quiz_data['explanation'] = line.replace('EXPLANATION:', '').strip()
            elif line.startswith('HINT:'):
                quiz_data['hint'] = line.replace('HINT:', '').strip()
            elif line.startswith(('a)', 'b)', 'c)')) and current_section == 'options':
                option = line[0]
                quiz_data['options'][option] = line[2:].strip()
                
        return quiz_data

    def conduct_quiz(self):
        """Conduct an interactive quiz with multiple attempts and hints"""
        # Get quiz from LLM
        quiz_response = self.quiz_chain.run(topic=self.topic)
        quiz_data = self.parse_quiz_response(quiz_response)
        
        # Display question and options
        print("\nQuiz Question:")
        print(quiz_data['question'])
        print("\nOptions:")
        for option, text in quiz_data['options'].items():
            print(f"{option}) {text}")
        
        # First attempt
        answer = input("\nYour answer (a/b/c): ").lower()
        
        if answer == quiz_data['correct_answer']:
            print("\nCorrect! Well done!")
            print("Explanation:", quiz_data['explanation'])
            return True
        else:
            print("\nThat's not quite right. Would you like a hint? (yes/no)")
            want_hint = input().lower()
            
            if want_hint == 'yes':
                print("\nHint:", quiz_data['hint'])
            
            # Second attempt
            print("\nTry one more time!")
            answer = input("Your answer (a/b/c): ").lower()
            
            if answer == quiz_data['correct_answer']:
                print("\nCorrect! You got it on the second try!")
                print("Explanation:", quiz_data['explanation'])
                return True
            else:
                print("\nIncorrect. The correct answer was:", quiz_data['correct_answer'])
                print("Explanation:", quiz_data['explanation'])
                return False

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
                
                while True:
                    # Step 4: Ask what they want to do next
                    print("\nWhat would you like to do next?")
                    print("1. See more examples")
                    print("2. Take a quiz")
                    print("3. End lesson")
                    
                    choice = input("Enter your choice (1/2/3): ")
                    
                    if choice == '1':
                        print("\nHere are some more examples...")
                        examples = self.example_chain.run(topic=self.topic)
                        print(examples)
                    elif choice == '2':
                        print("\nLet's test your understanding with a quiz!")
                        quiz_result = self.conduct_quiz()
                        
                        print("\nWould you like to:")
                        print("1. Continue learning")
                        print("2. End lesson")
                        
                        end_choice = input("Enter your choice (1/2): ")
                        if end_choice == '2':
                            print("Goodbye!")
                            return
                        break
                    elif choice == '3':
                        print("Goodbye!")
                        return
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")

# Create and run the teaching agent
agent = TeachingAgent()
agent.start_lesson()