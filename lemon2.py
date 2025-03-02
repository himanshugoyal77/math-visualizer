import os
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import streamlit as st
from matplotlib import pyplot as plt
import plotly.express as px
import io
import contextlib
from langchain_groq import ChatGroq

class TeachingAgent:
    def __init__(self):
        self.llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", return_messages=True)
        self.user_progress = {}
        self.current_user = None
        self.learning_style = None
        self.difficulty_level = "beginner"
        self.topic = None
        self.preferred_subjects = []
        self.attention_span = 30
        self.prior_knowledge = ""
        self.learning_goals = ""
        self.preferred_language = "English"
        self.time_availability = "1 hour per day"
        
        self.base_prompt_vars = [
            "topic", "learning_style", "difficulty_level", "chat_history", "human_input",
            "preferred_subjects", "attention_span", "prior_knowledge", "learning_goals",
            "preferred_language", "time_availability"
        ]
        
        self.explain_prompt = self._create_prompt_template("""
            You are explaining {topic} to a {learning_style} learner at {difficulty_level} level.
            The learner's preferred subjects are: {preferred_subjects}.
            Their attention span is {attention_span} minutes.
            They already know: {prior_knowledge}.
            Their learning goals are: {learning_goals}.
            They prefer learning in {preferred_language}.
            They can dedicate {time_availability} to learning.
            
            Keep the explanation clear, engaging, and tailored to their profile. If possible, add interactive elements like flowcharts or diagrams.
            
            Previous conversation:
            {chat_history}
            
            Current question or topic focus:
            {human_input}
            
            If a visualization (chart, diagram, or graph) would enhance understanding:
            1. Describe the visualization in one sentence
            2. Generate Python code using matplotlib to create it
            3. Make sure the code:
            - Includes 'import matplotlib.pyplot as plt'
            - Uses 'plt.figure(figsize=(10, 6))' for appropriate sizing
            - Does NOT include plt.show()
            4. Mention this is optional and can be run locally
            
            Format any code with triple backticks.
            
            Provide a thorough explanation:
            """
        )
        
        self.topic_validation_prompt = self._create_prompt_template("""
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
        
        self.example_prompt = self._create_prompt_template("""
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
        
        self.quiz_prompt = self._create_prompt_template("""
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
        
        self.visualization_prompt = self._create_prompt_template("""
            Generate visualization data for {topic} suitable for a {learning_style} learner at {difficulty_level} level.
            The learner's profile:
            - Preferred subjects: {preferred_subjects}
            - Attention span: {attention_span} minutes
            - Prior knowledge: {prior_knowledge}
            - Learning goals: {learning_goals}
            - Preferred language: {preferred_language}
            - Time availability: {time_availability}
            
            Previous conversation:
            {chat_history}
            
            Current focus:
            {human_input}
            
            Provide visualization data in the following format:
            CHART_TYPE: [line, bar, scatter, pie, flowchart, tree]
            TITLE: (visualization title)
            X_LABEL: (x-axis label if applicable)
            Y_LABEL: (y-axis label if applicable)
            DATA:
            (JSON format data for visualization)
            DESCRIPTION: (brief description of what the visualization shows)
            """
        )
        
        self.chains = {
            'visualization': LLMChain(llm=self.llm, prompt=self.visualization_prompt, memory=self.memory, verbose=True),
            'topic_validation': LLMChain(llm=self.llm, prompt=self.topic_validation_prompt, memory=self.memory, verbose=True),
            'explain': LLMChain(llm=self.llm, prompt=self.explain_prompt, memory=self.memory, verbose=True),
            'example': LLMChain(llm=self.llm, prompt=self.example_prompt, memory=self.memory, verbose=True),
            'quiz': LLMChain(llm=self.llm, prompt=self.quiz_prompt, memory=self.memory, verbose=True)
        }

    def _create_prompt_template(self, template):
        """Create a prompt template with common input variables."""
        return PromptTemplate(input_variables=self.base_prompt_vars, template=template)

    def extract_visualization(self, response):
        """Extract visualization description and code from response."""
        if '```python' in response:
            desc_part, code_part = response.split('```python')[0].strip(), response.split('```python')[1].split('```')[0].strip()
            return desc_part, code_part
        return response, None

    def parse_visualization_response(self, response):
        """Parse the LLM response into visualization data."""
        try:
            lines = response.strip().split('\n')
            visualization_data = {}
            current_key = None
            data_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line and not line.startswith(' '):
                    if current_key == 'DATA' and data_lines:
                        try:
                            visualization_data['DATA'] = json.loads('\n'.join(data_lines))
                        except:
                            visualization_data['DATA'] = '\n'.join(data_lines)
                        data_lines = []
                    
                    current_key, value = line.split(':', 1)
                    current_key = current_key.strip()
                    value = value.strip()
                    
                    if current_key != 'DATA':
                        visualization_data[current_key] = value
                else:
                    if current_key == 'DATA':
                        data_lines.append(line)
            
            if current_key == 'DATA' and data_lines:
                try:
                    visualization_data['DATA'] = json.loads('\n'.join(data_lines))
                except:
                    visualization_data['DATA'] = '\n'.join(data_lines)
            
            return visualization_data
            
        except Exception as e:
            st.error(f"Error parsing visualization response: {e}")
            return None

    def get_explanation(self, topic, user_input):
        """Get an explanation with visualization for the given topic."""
        try:
            explanation = self.chains['explain'].run(
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
            
            visualization_response = self.chains['visualization'].run(
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
            
            visualization_data = self.parse_visualization_response(visualization_response)
            if visualization_data:
                st.write("### Explanation")
                st.write(explanation.split('```python')[0])
            else:
                st.write(explanation.split('```python')[0])
            
            return explanation
            
        except Exception as e:
            st.error(f"Error getting explanation: {e}")
            return "I'm having trouble generating an explanation. Let's try again."

    def start_lesson(self):
        """Start a lesson with enhanced personalization and visualization."""
        st.write("### Start a New Lesson")
        
        topic_container = st.container()
        with topic_container:
            self.topic = st.text_input("What would you like to learn about?")
            
            if self.topic:
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                st.success(f"Ready to learn about {self.topic}!")
                
                self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input", return_messages=True)
                for chain_name in self.chains:
                    self.chains[chain_name].memory = self.memory
                
                user_input = st.text_input(f"What specifically would you like to know about {self.topic}?")
                
                if user_input:
                    explanation = self.get_explanation(self.topic, user_input)
                    explanation_text, vis_code = self.extract_visualization(explanation)
                    
                    if vis_code:
                        st.subheader("📊 Interactive Visualization")
                        try:
                            exec(vis_code)
                            st.pyplot(plt)
                            plt.close()
                            
                            st.write("Customize Visualization:")
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_size = st.slider("Figure Size", 6, 15, 10)
                            with col2:
                                theme = st.selectbox("Color Theme", ["default", "dark_background", "seaborn"])
                            
                            if st.button("Update Visualization"):
                                plt.style.use(theme)
                                plt.figure(figsize=(fig_size, fig_size*0.6))
                                exec(vis_code)
                                st.pyplot(plt)
                                plt.close()
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
        
                    self._handle_lesson_flow()

    def create_or_select_user_profile(self):
        """Create or select a user profile."""
        st.header("User Profile")
        
        tab1, tab2 = st.tabs(["Create New Profile", "Select Existing Profile"])
        
        with tab1:
            self._create_user_profile()
        
        with tab2:
            self._select_user_profile()

    def _create_user_profile(self):
        """Create a new user profile."""
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
        
        learning_style = st.radio("Choose your learning style:", options=["Visual", "Auditory", "Kinesthetic"])
        self.learning_style = learning_style.lower()
        
        difficulty_level = st.radio("Choose your difficulty level:", options=["Beginner", "Intermediate", "Advanced"])
        self.difficulty_level = difficulty_level.lower()
        
        subjects = st.text_input("Enter preferred subjects (comma-separated):")
        if subjects:
            self.preferred_subjects = [s.strip() for s in subjects.split(",")]
        
        attention_span = st.number_input("Enter attention span (minutes):", min_value=5, max_value=120, value=30)
        self.attention_span = attention_span
        
        prior_knowledge = st.text_area("What do you already know about your subjects of interest?")
        self.prior_knowledge = prior_knowledge
        
        learning_goals = st.text_area("What are your learning goals?")
        self.learning_goals = learning_goals
        
        preferred_language = st.text_input("Enter preferred language:", value="English")
        self.preferred_language = preferred_language
        
        time_availability = st.text_input("How much time can you dedicate to learning each day?", value="1 hour per day")
        self.time_availability = time_availability
        
        if st.button("Create Profile"):
            if username:
                self.user_progress[username].update({
                    "learning_style": self.learning_style,
                    "difficulty_level": self.difficulty_level,
                    "preferred_subjects": self.preferred_subjects,
                    "attention_span": self.attention_span,
                    "prior_knowledge": self.prior_knowledge,
                    "learning_goals": self.learning_goals,
                    "preferred_language": self.preferred_language,
                    "time_availability": self.time_availability
                })
                self.save_progress()
                st.success(f"Profile created for {username}!")
                st.session_state.page = "topics"
                st.rerun()
            else:
                st.error("Username cannot be empty.")

    def _select_user_profile(self):
        """Select an existing user profile."""
        existing_profiles = [f.split('_')[0] for f in os.listdir() if f.endswith('_progress.json')]
        
        if not existing_profiles:
            st.warning("No existing profiles found. Please create a new profile.")
            return
        
        username = st.selectbox("Select your profile:", existing_profiles)
        
        if st.button("Load Profile"):
            if username in self.user_progress or os.path.exists(f"{username}_progress.json"):
                self.current_user = username
                self.load_progress()
                
                profile_data = self.user_progress[username]
                self.learning_style = profile_data["learning_style"]
                self.difficulty_level = profile_data["difficulty_level"]
                self.preferred_subjects = profile_data["preferred_subjects"]
                self.attention_span = profile_data["attention_span"]
                self.prior_knowledge = profile_data["prior_knowledge"]
                self.learning_goals = profile_data["learning_goals"]
                self.preferred_language = profile_data["preferred_language"]
                self.time_availability = profile_data["time_availability"]
                
                st.success(f"Welcome back, {username}!")
                self.display_profile_summary()
                
                st.session_state.page = "topics"
                st.rerun()
            else:
                st.error("User not found. Please create a new profile.")

    def display_profile_summary(self):
        """Display a summary of the user's profile."""
        st.subheader("Profile Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Learning Style:** {self.learning_style.title()}")
            st.write(f"**Difficulty Level:** {self.difficulty_level.title()}")
            st.write(f"**Preferred Subjects:** {', '.join(self.preferred_subjects)}")
            st.write(f"**Attention Span:** {self.attention_span} minutes")
        
        with col2:
            st.write(f"**Preferred Language:** {self.preferred_language}")
            st.write(f"**Time Availability:** {self.time_availability}")
            st.write(f"**Learning Goals:** {self.learning_goals[:100]}...")

    def save_progress(self):
        """Save user progress to a file."""
        if self.current_user:
            try:
                with open(f"{self.current_user}_progress.json", "w") as f:
                    json.dump(self.user_progress, f)
            except Exception as e:
                st.error(f"Error saving progress: {e}")

    def load_progress(self):
        """Load user progress from a file."""
        if self.current_user:
            try:
                with open(f"{self.current_user}_progress.json", "r") as f:
                    self.user_progress = json.load(f)
            except FileNotFoundError:
                self.user_progress = {}
            except Exception as e:
                st.error(f"Error loading progress: {e}")
                self.user_progress = {}

    def _handle_lesson_flow(self):
        """Handle the main lesson flow after initial explanation."""
        understanding = st.radio("Do you understand this explanation?", options=["Yes", "No", "I have a question"])
        
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
        """Handle the next steps after understanding check."""
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

    def get_examples(self, topic, user_input):
        """Get examples for the given topic."""
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
        """Conduct an interactive quiz with multiple attempts and hints."""
        try:
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
            
            st.write("### Quiz Question")
            st.write(quiz_data['question'])
            st.write("### Options")
            for option, text in quiz_data['options'].items():
                st.write(f"{option}) {text}")
            
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
        """Parse the raw quiz response into a structured format."""
        try:
            quiz_data = {
                "question": "",
                "options": {},
                "correct_answer": "",
                "explanation": "",
                "hint": ""
            }
            
            current_section = None
            
            for line in quiz_response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
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
                    if current_section == "options" and ')' in line:
                        option, text = line.split(')', 1)
                        option = option.strip().lower()
                        quiz_data["options"][option] = text.strip()
                    elif current_section in ["explanation", "hint"]:
                        quiz_data[current_section] += " " + line
            
            if not all([quiz_data["question"], quiz_data["options"], quiz_data["correct_answer"]]):
                return None
                
            return quiz_data
            
        except Exception as e:
            st.error(f"Error parsing quiz response: {e}")
            return None

def main():
    st.set_page_config(page_title="Personalized Tutor AI", layout="wide")
    
    if 'page' not in st.session_state:
        st.session_state.page = "profile"
    if 'agent' not in st.session_state:
        st.session_state.agent = TeachingAgent()
    
    st.title("Personalized Tutor AI")
    
    if st.session_state.page == "profile":
        st.session_state.agent.create_or_select_user_profile()
    elif st.session_state.page == "topics":
        if st.button("← Back to Profile"):
            st.session_state.page = "profile"
            st.rerun()
        
        with st.sidebar:
            st.session_state.agent.display_profile_summary()
        
        st.session_state.agent.start_lesson()

if __name__ == "__main__":
    main()