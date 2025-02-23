import os
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import streamlit as st

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
        self.preferred_subjects = []
        self.attention_span = 30  # Default attention span in minutes
        self.prior_knowledge = ""
        self.learning_goals = ""
        self.preferred_language = "English"
        self.time_availability = "1 hour per day"
        
        # Update prompts to include more personalization
        self.explain_prompt = PromptTemplate(
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input", 
                            "preferred_subjects", "attention_span", "prior_knowledge", "learning_goals", 
                            "preferred_language", "time_availability"],
            template="""
            You are explaining {topic} to a {learning_style} learner at {difficulty_level} level.
            The learner's preferred subjects are: {preferred_subjects}.
            Their attention span is {attention_span} minutes.
            They already know: {prior_knowledge}.
            Their learning goals are: {learning_goals}.
            They prefer learning in {preferred_language}.
            They can dedicate {time_availability} to learning.
            
            Keep the explanation clear, engaging, and tailored to their profile.
            
            Previous conversation:
            {chat_history}
            
            Current question or topic focus:
            {human_input}
            
            Provide a thorough explanation:
            """
        )
        
        # Update other prompts similarly
        self.topic_validation_prompt = PromptTemplate(
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input", 
                            "preferred_subjects", "attention_span", "prior_knowledge", "learning_goals", 
                            "preferred_language", "time_availability"],
            template="""
            Is {topic} a valid educational topic for a {learning_style} learner at {difficulty_level} level?
            Consider the learner's profile:
            - Preferred subjects: {preferred_subjects}
            - Attention span: {attention_span} minutes
            - Prior knowledge: {prior_knowledge}
            - Learning goals: {learning_goals}
            - Preferred language: {preferred_language}
            - Time availability: {time_availability}
            
            Respond with 'VALID: topic' or 'INVALID: reason'.
            """
        )
        
        self.example_prompt = PromptTemplate(
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input", 
                             "preferred_subjects", "attention_span", "prior_knowledge", "learning_goals", 
                             "preferred_language", "time_availability"],
            template="""
            Give 2-3 practical examples of {topic} suited for {learning_style} learners.
            Examples should be at {difficulty_level} level and tailored to:
            - Preferred subjects: {preferred_subjects}
            - Attention span: {attention_span} minutes
            - Prior knowledge: {prior_knowledge}
            - Learning goals: {learning_goals}
            - Preferred language: {preferred_language}
            - Time availability: {time_availability}
            
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
            input_variables=["topic", "learning_style", "difficulty_level", "chat_history", "human_input", 
                            "preferred_subjects", "attention_span", "prior_knowledge", "learning_goals", 
                            "preferred_language", "time_availability"],
            template="""
            Create a multiple choice question about {topic} for {difficulty_level} level.
            The question should be suited for {learning_style} learners and tailored to:
            - Preferred subjects: {preferred_subjects}
            - Attention span: {attention_span} minutes
            - Prior knowledge: {prior_knowledge}
            - Learning goals: {learning_goals}
            - Preferred language: {preferred_language}
            - Time availability: {time_availability}
            
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

    def create_user_profile(self):
        """Create a new user profile with enhanced attributes"""
        try:
            username = st.text_input("Enter a username:")
            if username:
                self.current_user = username
                self.user_progress[username] = {
                    "topics_learned": [],
                    "quiz_scores": {},
                    "last_session": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "learning_style": None,
                    "difficulty_level": "beginner",
                    "preferred_subjects": [],
                    "attention_span": 30,
                    "prior_knowledge": "",
                    "learning_goals": "",
                    "preferred_language": "English",
                    "time_availability": "1 hour per day"
                }
                self.set_learning_style()
                self.set_difficulty_level()
                self.set_preferred_subjects()
                self.set_attention_span()
                self.set_prior_knowledge()
                self.set_learning_goals()
                self.set_preferred_language()
                self.set_time_availability()
                self.save_progress()
                st.success(f"Profile created for {username}.")
            else:
                st.error("Username cannot be empty.")
        except Exception as e:
            st.error(f"Error creating user profile: {e}")

    def set_learning_style(self):
        """Set the user's preferred learning style"""
        try:
            st.write("\nWhat is your preferred learning style?")
            learning_style = st.radio(
                "Choose your learning style:",
                options=["Visual", "Auditory", "Kinesthetic"]
            )
            self.learning_style = learning_style.lower()
            if self.current_user:
                self.user_progress[self.current_user]["learning_style"] = self.learning_style
        except Exception as e:
            st.error(f"Error setting learning style: {e}")

    def set_difficulty_level(self):
        """Set the user's preferred difficulty level"""
        try:
            st.write("\nWhat is your preferred difficulty level?")
            difficulty_level = st.radio(
                "Choose your difficulty level:",
                options=["Beginner", "Intermediate", "Advanced"]
            )
            self.difficulty_level = difficulty_level.lower()
            if self.current_user:
                self.user_progress[self.current_user]["difficulty_level"] = self.difficulty_level
        except Exception as e:
            st.error(f"Error setting difficulty level: {e}")

    def set_preferred_subjects(self):
        """Set the user's preferred subjects"""
        try:
            st.write("\nWhat subjects are you most interested in? (comma-separated list)")
            subjects = st.text_input("Enter subjects:")
            if subjects:
                self.preferred_subjects = [s.strip() for s in subjects.split(",")]
                if self.current_user:
                    self.user_progress[self.current_user]["preferred_subjects"] = self.preferred_subjects
        except Exception as e:
            st.error(f"Error setting preferred subjects: {e}")

    def set_attention_span(self):
        """Set the user's attention span"""
        try:
            st.write("\nHow long can you typically focus on a topic? (in minutes)")
            attention_span = st.number_input("Enter attention span:", min_value=5, max_value=120, value=30)
            self.attention_span = attention_span
            if self.current_user:
                self.user_progress[self.current_user]["attention_span"] = self.attention_span
        except Exception as e:
            st.error(f"Error setting attention span: {e}")

    def set_prior_knowledge(self):
        """Set the user's prior knowledge"""
        try:
            st.write("\nWhat do you already know about the subject?")
            prior_knowledge = st.text_area("Enter prior knowledge:")
            if prior_knowledge:
                self.prior_knowledge = prior_knowledge
                if self.current_user:
                    self.user_progress[self.current_user]["prior_knowledge"] = self.prior_knowledge
        except Exception as e:
            st.error(f"Error setting prior knowledge: {e}")

    def set_learning_goals(self):
        """Set the user's learning goals"""
        try:
            st.write("\nWhat are your learning goals?")
            learning_goals = st.text_area("Enter learning goals:")
            if learning_goals:
                self.learning_goals = learning_goals
                if self.current_user:
                    self.user_progress[self.current_user]["learning_goals"] = self.learning_goals
        except Exception as e:
            st.error(f"Error setting learning goals: {e}")

    def set_preferred_language(self):
        """Set the user's preferred language"""
        try:
            st.write("\nWhat is your preferred language for learning?")
            preferred_language = st.text_input("Enter preferred language:")
            if preferred_language:
                self.preferred_language = preferred_language
                if self.current_user:
                    self.user_progress[self.current_user]["preferred_language"] = self.preferred_language
        except Exception as e:
            st.error(f"Error setting preferred language: {e}")

    def set_time_availability(self):
        """Set the user's time availability"""
        try:
            st.write("\nHow much time can you dedicate to learning each day/week?")
            time_availability = st.text_input("Enter time availability:")
            if time_availability:
                self.time_availability = time_availability
                if self.current_user:
                    self.user_progress[self.current_user]["time_availability"] = self.time_availability
        except Exception as e:
            st.error(f"Error setting time availability: {e}")

    def select_user_profile(self):
        """Select an existing user profile and load all attributes"""
        try:
            username = st.text_input("Enter your username:")
            if username in self.user_progress:
                self.current_user = username
                self.load_progress()
                self.learning_style = self.user_progress[username]["learning_style"]
                self.difficulty_level = self.user_progress[username]["difficulty_level"]
                self.preferred_subjects = self.user_progress[username]["preferred_subjects"]
                self.attention_span = self.user_progress[username]["attention_span"]
                self.prior_knowledge = self.user_progress[username]["prior_knowledge"]
                self.learning_goals = self.user_progress[username]["learning_goals"]
                self.preferred_language = self.user_progress[username]["preferred_language"]
                self.time_availability = self.user_progress[username]["time_availability"]
                st.success(f"Welcome back, {username}!")
            else:
                st.error("User not found. Please create a new profile.")
        except Exception as e:
            st.error(f"Error selecting user profile: {e}")

    def save_progress(self):
        """Save user progress to a file"""
        if self.current_user:
            try:
                with open(f"{self.current_user}_progress.json", "w") as f:
                    json.dump(self.user_progress, f)
            except Exception as e:
                st.error(f"Error saving progress: {e}")

    def load_progress(self):
        """Load user progress from a file"""
        if self.current_user:
            try:
                with open(f"{self.current_user}_progress.json", "r") as f:
                    self.user_progress = json.load(f)
            except FileNotFoundError:
                self.user_progress = {}
            except Exception as e:
                st.error(f"Error loading progress: {e}")
                self.user_progress = {}

    def start_lesson(self):
        """Start a lesson with enhanced personalization"""
        st.write("### Start a New Lesson")
        self.topic = st.text_input("What would you like to learn about?")
        
        if self.topic:
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
            user_input = st.text_input(f"What specifically would you like to know about {self.topic}?")
            
            if user_input:
                # Get and display explanation
                explanation = self.get_explanation(self.topic, user_input)
                st.write("### Explanation")
                st.write(explanation)
                
                # Continue with the rest of the lesson flow
                self._handle_lesson_flow()

    def _handle_lesson_flow(self):
        """Handle the main lesson flow after initial explanation"""
        understanding = st.radio(
            "Do you understand this explanation?",
            options=["Yes", "No", "I have a question"]
        )
        
        if understanding == "No":
            st.write("Let me explain it differently...")
            user_input = "Please explain this again differently"
            explanation = self.get_explanation(self.topic, user_input)
            st.write(explanation)
        elif understanding == "I have a question":
            user_input = st.text_input("What's your question?")
            if user_input:
                explanation = self.get_explanation(self.topic, user_input)
                st.write(explanation)
        elif understanding == "Yes":
            st.write("Great! Let's move on to examples and quizzes.")
            self._handle_next_steps()

    def _handle_next_steps(self):
        """Handle the next steps after understanding check"""
        option = st.radio(
            "What would you like to do next?",
            options=["See examples", "Ask another question", "Take a quiz", "Start new topic", "End session"]
        )
        
        if option == "See examples":
            examples = self.get_examples(self.topic, "Provide examples")
            st.write("### Examples")
            st.write(examples)
        elif option == "Ask another question":
            user_input = st.text_input("What's your question?")
            if user_input:
                explanation = self.get_explanation(self.topic, user_input)
                st.write(explanation)
        elif option == "Take a quiz":
            self.conduct_quiz(self.topic, "Generate a quiz")
        elif option == "Start new topic":
            self.topic = None
            st.experimental_rerun()
        elif option == "End session":
            st.write("Goodbye!")
            st.stop()

    def get_explanation(self, topic, user_input):
        """Get an explanation for the given topic"""
        try:
            return self.chains['explain'].run(
                topic=topic,
                learning_style=self.learning_style,
                difficulty_level=self.difficulty_level,
                human_input=user_input,
                preferred_subjects=self.preferred_subjects,
                attention_span=self.attention_span,
                prior_knowledge=self.prior_knowledge,
                learning_goals=self.learning_goals,
                preferred_language=self.preferred_language,
                time_availability=self.time_availability
            )
        except Exception as e:
            st.error(f"Error getting explanation: {e}")
            return "I'm having trouble generating an explanation. Let's try again."

    def get_examples(self, topic, user_input):
        """Get examples for the given topic"""
        try:
            return self.chains['example'].run(
                topic=topic,
                learning_style=self.learning_style,
                difficulty_level=self.difficulty_level,
                human_input=user_input,
                preferred_subjects=self.preferred_subjects,
                attention_span=self.attention_span,
                prior_knowledge=self.prior_knowledge,
                learning_goals=self.learning_goals,
                preferred_language=self.preferred_language,
                time_availability=self.time_availability
            )
        except Exception as e:
            st.error(f"Error getting examples: {e}")
            return "I'm having trouble generating examples. Let's try again."

    def conduct_quiz(self, topic, user_input):
        """Conduct an interactive quiz with multiple attempts and hints"""
        try:
            # Get quiz from LLM
            quiz_response = self.chains['quiz'].run(
                topic=topic,
                learning_style=self.learning_style,
                difficulty_level=self.difficulty_level,
                human_input=user_input,
                preferred_subjects=self.preferred_subjects,
                attention_span=self.attention_span,
                prior_knowledge=self.prior_knowledge,
                learning_goals=self.learning_goals,
                preferred_language=self.preferred_language,
                time_availability=self.time_availability
            )
            
            quiz_data = self.parse_quiz_response(quiz_response)
            if not quiz_data:
                st.error("Error generating quiz. Please try again.")
                return False
            
            # Display question and options
            st.write("### Quiz Question")
            st.write(quiz_data['question'])
            st.write("### Options")
            for option, text in quiz_data['options'].items():
                st.write(f"{option}) {text}")
            
            # First attempt
            answer = st.radio("Your answer:", options=["a", "b", "c"])
            
            if answer == quiz_data['correct_answer']:
                st.success("Correct! Well done!")
                st.write("### Explanation")
                st.write(quiz_data['explanation'])
                return True
            else:
                want_hint = st.radio("That's not quite right. Would you like a hint?", options=["Yes", "No"])
                
                if want_hint == "Yes":
                    st.write("### Hint")
                    st.write(quiz_data['hint'])
                
                # Second attempt
                answer = st.radio("Try one more time! Your answer:", options=["a", "b", "c"])
                
                if answer == quiz_data['correct_answer']:
                    st.success("Correct! You got it on the second try!")
                    st.write("### Explanation")
                    st.write(quiz_data['explanation'])
                    return True
                else:
                    st.error(f"Incorrect. The correct answer was: {quiz_data['correct_answer']}")
                    st.write("### Explanation")
                    st.write(quiz_data['explanation'])
                    return False
        except Exception as e:
            st.error(f"Error conducting quiz: {e}")
            return False

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
            st.error(f"Error parsing quiz response: {e}")
            return None

# Streamlit App
def main():
    st.title("Personalized Tutor AI")
    
    agent = TeachingAgent()
    
    # User profile management
    st.sidebar.title("User Profile")
    profile_choice = st.sidebar.radio(
        "Choose an option:",
        options=["Create new profile", "Select existing profile"]
    )
    
    if profile_choice == "Create new profile":
        agent.create_user_profile()
    elif profile_choice == "Select existing profile":
        agent.select_user_profile()
    
    # Start lesson
    if agent.current_user:
        agent.start_lesson()

if __name__ == "__main__":
    main()