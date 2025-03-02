import os
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

class TeachingAgent:
    def __init__(self):
        # Initialize LLM and memory
        self.llm = ChatOpenAI(
            model='google/gemini-2.0-flash-001',
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_KEY")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            input_key="human_input",
            return_messages=True
        )
        
        # Initialize user-related attributes
        self.user_progress = {}
        self.current_user = None
        self.learning_style = None
        self.difficulty_level = "beginner"
        self.topic = None
        
        # Update the explain prompt to properly use the input variables
        self.explain_prompt = PromptTemplate(
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input"],
            template="""
            You are explaining {topic} to a {learning_style} learner at {difficulty_level} level.
            Keep the explanation clear and engaging.
            
            Previous conversation:
            {chat_history}
            
            Current question or topic focus:
            {human_input}
            
            Provide a thorough explanation:
            """
        )
        
        # Update other prompts to include the same input variables
        self.topic_validation_prompt = PromptTemplate(
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input"],
            template="""
            Is {topic} a valid educational topic for a {learning_style} learner at {difficulty_level} level?
            Consider the following conversation context:
            {chat_history}
            
            Respond with 'VALID: topic' or 'INVALID: reason'.
            """
        )
        
        self.example_prompt = PromptTemplate(
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input"],
            template="""
            Give 2-3 practical examples of {topic} suited for {learning_style} learners.
            Examples should be at {difficulty_level} level.
            
            Previous conversation:
            {chat_history}
            
            Current question or topic focus:
            {human_input}
            
            Examples:
            1) (example1)
            2) (example2)
            3) (example3)
            """
        )
        
        self.quiz_prompt = PromptTemplate(
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input"],
            template="""
            Create a multiple choice question about {topic} for {difficulty_level} level.
            The question should be suited for {learning_style} learners.
            
            Previous conversation:
            {chat_history}
            
            Current question or topic focus:
            {human_input}
            
            Format:
            QUESTION: (question)
            OPTIONS:
            a) (option1)
            b) (option2)
            c) (option3)
            CORRECT_ANSWER: (letter)
            EXPLANATION: (explanation)
            HINT: (hint)
            """
        )
        
        # Initialize all chains with memory and consistent input variables
        self.chains = {
            'topic_validation': LLMChain(
                llm=self.llm,
                prompt=self.topic_validation_prompt,
                memory=self.memory,
                verbose=True
            ),
            'explain': LLMChain(
                llm=self.llm,
                prompt=self.explain_prompt,
                memory=self.memory,
                verbose=True
            ),
            'example': LLMChain(
                llm=self.llm,
                prompt=self.example_prompt,
                memory=self.memory,
                verbose=True
            ),
            'quiz': LLMChain(
                llm=self.llm,
                prompt=self.quiz_prompt,
                memory=self.memory,
                verbose=True
            )
        }

    def validate_topic(self, topic):
        """Validate if the topic is appropriate for teaching"""
        try:
            response = self.chains['topic_validation'].run(
                topic=topic,
                learning_style=self.learning_style,
                difficulty_level=self.difficulty_level,
                human_input="Validating topic"
            )
            if response.startswith("VALID:"):
                return True, response[6:].strip()
            return False, response[8:].strip()
        except Exception as e:
            print(f"Error validating topic: {e}")
            return False, "Error validating topic"

    def get_topic(self):
        """Get and validate topic from user"""
        while True:
            try:
                print("\nWhat would you like to learn about?")
                print("(Type 'quit' to exit)")
                topic = input("Enter topic: ").strip()
                
                if topic.lower() == 'quit':
                    return None
                    
                is_valid, message = self.validate_topic(topic)
                if is_valid:
                    print(f"\nGreat! Let's learn about {message}")
                    return message
                else:
                    print(f"\nSorry, {message}")
                    print("Please try entering a more specific or appropriate topic.")
            except Exception as e:
                print(f"Error getting topic: {e}")
                return None

    def get_explanation(self, topic, user_input):
        """Get an explanation for the given topic"""
        try:
            return self.chains['explain'].run(
                topic=topic,
                learning_style=self.learning_style,
                difficulty_level=self.difficulty_level,
                human_input=user_input
            )
        except Exception as e:
            print(f"Error getting explanation: {e}")
            return "I'm having trouble generating an explanation. Let's try again."

    def get_examples(self, topic, user_input):
        """Get examples for the given topic"""
        try:
            return self.chains['example'].run(
                topic=topic,
                learning_style=self.learning_style,
                difficulty_level=self.difficulty_level,
                human_input=user_input
            )
        except Exception as e:
            print(f"Error getting examples: {e}")
            return "I'm having trouble generating examples. Let's try again."

    def conduct_quiz(self, topic, user_input):
        """Conduct an interactive quiz with multiple attempts and hints"""
        try:
            # Get quiz from LLM
            quiz_response = self.chains['quiz'].run(
                topic=topic,
                learning_style=self.learning_style,
                difficulty_level=self.difficulty_level,
                human_input=user_input
            )
            
            quiz_data = self.parse_quiz_response(quiz_response)
            if not quiz_data:
                print("Error generating quiz. Please try again.")
                return False
            
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
        except Exception as e:
            print(f"Error conducting quiz: {e}")
            return False

    def start_lesson(self):
        while True:
            # Get the topic from user
            self.topic = input("\nWhat would you like to learn about? ").strip()
            print("Topic:", self.topic)
            if not self.topic:
                print("Goodbye!")
                break
            
            # Reset memory for new topic
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="human_input",
                return_messages=True
            )
            
            # Update all chains with new memory
            for chain_name in self.chains:
                self.chains[chain_name].memory = self.memory
            
            # Get initial user input about what they want to know
            print(f"\nWhat specifically would you like to know about {self.topic}?")
            user_input = input("Your question: ").strip()
            
            # Get and display explanation
            explanation = self.get_explanation(self.topic, user_input+ ' ' + self.topic)
            print("\nLet me explain:", self.topic)
            print(explanation)
            
            # Continue with the rest of the lesson flow
            self._handle_lesson_flow()

    def _handle_lesson_flow(self):
        """Handle the main lesson flow after initial explanation"""
        while True:
            understanding = input("\nDo you understand this explanation? (yes/no/question): ").lower()
            
            if understanding == 'no':
                print("\nLet me explain it differently...")
                user_input = "Please explain this again differently"
                explanation = self.get_explanation(self.topic, user_input)
                print(explanation)
            elif understanding == 'question':
                user_input = input("\nWhat's your question? ")
                explanation = self.get_explanation(self.topic, user_input)
                print(explanation)
            elif understanding == 'yes':
                break
            else:
                print("Please answer with 'yes', 'no', or 'question'")
        
        # Continue with examples and quiz
        self._handle_next_steps()

    def _handle_next_steps(self):
        """Handle the next steps after understanding check"""
        while True:
            print("\nWhat would you like to do next?")
            print("1. See examples")
            print("2. Ask another question")
            print("3. Take a quiz")
            print("4. Start new topic")
            print("5. End session")
            
            choice = input("Enter your choice (1/2/3/4/5): ")
            
            if choice == '1':
                examples = self.get_examples(self.topic, "Provide examples")
                print("\nHere are some examples...")
                print(examples)
            elif choice == '2':
                user_input = input("\nWhat's your question? ")
                explanation = self.get_explanation(self.topic, user_input + self.topic)
                print(explanation)
            elif choice == '3':
                self.conduct_quiz(self.topic, "Generate a quiz")
            elif choice == '4':
                return
            elif choice == '5':
                print("Goodbye!")
                exit()
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

    def save_progress(self):
        """Save user progress to a file"""
        if self.current_user:
            try:
                with open(f"{self.current_user}_progress.json", "w") as f:
                    json.dump(self.user_progress, f)
            except Exception as e:
                print(f"Error saving progress: {e}")

    def load_progress(self):
        """Load user progress from a file"""
        if self.current_user:
            try:
                with open(f"{self.current_user}_progress.json", "r") as f:
                    self.user_progress = json.load(f)
            except FileNotFoundError:
                self.user_progress = {}
            except Exception as e:
                print(f"Error loading progress: {e}")
                self.user_progress = {}

    def create_user_profile(self):
        """Create a new user profile"""
        try:
            username = input("Enter a username: ").strip()
            if username:
                self.current_user = username
                self.user_progress[username] = {
                    "topics_learned": [],
                    "quiz_scores": {},
                    "last_session": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "learning_style": None,
                    "difficulty_level": "beginner"
                }
                self.set_learning_style()
                self.set_difficulty_level()
                self.save_progress()
                print(f"Profile created for {username}.")
            else:
                print("Username cannot be empty.")
        except Exception as e:
            print(f"Error creating user profile: {e}")

    def set_learning_style(self):
        """Set the user's preferred learning style"""
        try:
            print("\nWhat is your preferred learning style?")
            print("1. Visual (learn best with images, diagrams, and charts)")
            print("2. Auditory (learn best with spoken explanations and discussions)")
            print("3. Kinesthetic (learn best with hands-on activities and examples)")
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice == '1':
                self.learning_style = "visual"
            elif choice == '2':
                self.learning_style = "auditory"
            elif choice == '3':
                self.learning_style = "kinesthetic"
            else:
                print("Invalid choice. Defaulting to visual.")
                self.learning_style = "visual"
            if self.current_user:
                self.user_progress[self.current_user]["learning_style"] = self.learning_style
        except Exception as e:
            print(f"Error setting learning style: {e}")
            self.learning_style = "visual"

    def set_difficulty_level(self):
        """Set the user's preferred difficulty level"""
        try:
            print("\nWhat is your preferred difficulty level?")
            print("1. Beginner")
            print("2. Intermediate")
            print("3. Advanced")
            choice = input("Enter your choice (1/2/3): ").strip()
            if choice == '1':
                self.difficulty_level = "beginner"
            elif choice == '2':
                self.difficulty_level = "intermediate"
            elif choice == '3':
                self.difficulty_level = "advanced"
            else:
                print("Invalid choice. Defaulting to beginner.")
                self.difficulty_level = "beginner"
            if self.current_user:
                self.user_progress[self.current_user]["difficulty_level"] = self.difficulty_level
        except Exception as e:
            print(f"Error setting difficulty level: {e}")
            self.difficulty_level = "beginner"

    def select_user_profile(self):
        """Select an existing user profile"""
        try:
            username = input("Enter your username: ").strip()
            if username in self.user_progress:
                self.current_user = username
                self.load_progress()
                self.learning_style = self.user_progress[username]["learning_style"]
                self.difficulty_level = self.user_progress[username]["difficulty_level"]
                print(f"Welcome back, {username}!")
            else:
                print("User not found. Please create a new profile.")
        except Exception as e:
            print(f"Error selecting user profile: {e}")

    def parse_quiz_response(self, quiz_response):
        """Parse the raw quiz response into a structured format"""
        try:
            quiz_data = {
                "question": "",
                "options": {},
                "correct_answer": "",
                "explanation": "",
                "hint": ""
            }
            
            current_section = None
            
            # Split the response into lines and process each line
            for line in quiz_response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Detect section headers
                if line.startswith("QUESTION:"):
                    current_section = "question"
                    quiz_data["question"] = line[len("QUESTION:"):].strip()
                elif line.startswith("OPTIONS:"):
                    current_section = "options"
                elif line.startswith("CORRECT_ANSWER:"):
                    current_section = "correct_answer"
                    quiz_data["correct_answer"] = line[len("CORRECT_ANSWER:"):].strip().lower()
                elif line.startswith("EXPLANATION:"):
                    current_section = "explanation"
                    quiz_data["explanation"] = line[len("EXPLANATION:"):].strip()
                elif line.startswith("HINT:"):
                    current_section = "hint"
                    quiz_data["hint"] = line[len("HINT:"):].strip()
                else:
                    # Handle options and other content
                    if current_section == "options" and ')' in line:
                        option, text = line.split(')', 1)
                        option = option.strip().lower()
                        quiz_data["options"][option] = text.strip()
                    elif current_section in ["explanation", "hint"]:
                        # Append multi-line explanations/hints
                        quiz_data[current_section] += " " + line
            
            # Basic validation
            if not all([quiz_data["question"], quiz_data["options"], quiz_data["correct_answer"]]):
                return None
                
            return quiz_data
            
        except Exception as e:
            print(f"Error parsing quiz response: {e}")
            return None
        
if __name__ == "__main__":
    agent = TeachingAgent()

    # User profile management
    print("Welcome to the Personalized Tutor AI!")
    print("1. Create new profile")
    print("2. Select existing profile")
    profile_choice = input("Enter your choice (1/2): ")

    if profile_choice == '1':
        agent.create_user_profile()
    elif profile_choice == '2':
        agent.select_user_profile()
    else:
        print("Invalid choice. Exiting.")
        exit()

    agent.start_lesson()