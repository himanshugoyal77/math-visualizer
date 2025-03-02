from dataclasses import dataclass
from enum import Enum
from typing import Optional
from test import TeachingAgent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class LearningStyle(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    READING_WRITING = "reading_writing"
    KINESTHETIC = "kinesthetic"

@dataclass
class StudentProfile:
    grade_level: int
    difficulty_level: DifficultyLevel
    learning_style: LearningStyle
    preferred_subjects: list[str]
    attention_span: int  # in minutes
    prior_knowledge: Optional[str] = None

class PersonalizedTeachingAgent(TeachingAgent):
    def __init__(self):
        super().__init__()
        self.student_profile = None
        self.topic = None
        self.curriculum = {'current_topic': None}
        
        # Initialize additional prompt templates
        self.explain_prompt = PromptTemplate(
            input_variables=["topic", "human_input", "profile"],
            template="""
            Explain {topic} in response to: {human_input}
            
            Student profile:
            {profile}
            
            Tailor the explanation to match their learning style and keep within their attention span.
            """
        )
        
        self.example_prompt = PromptTemplate(
            input_variables=["topic", "profile"],
            template="""
            Provide practical examples of {topic} for a student with the following profile:
            {profile}
            
            Make examples relatable and appropriate for their grade level and learning style.
            """
        )
        
        # Initialize chains dictionary
        self.chains = {
            'explain': LLMChain(
                llm=self.llm,
                prompt=self.explain_prompt,
                memory=self.memory
            ),
            'example': LLMChain(
                llm=self.llm,
                prompt=self.example_prompt,
                memory=self.memory
            ),
            'quiz': self.personalized_quiz_chain  # Reuse existing quiz chain
        }

    def _interactive_learning_loop(self, topic: str):
        """Manage interactive learning session"""
        while True:
            print("\nLearning Options:")
            print("1. Ask a question")
            print("2. See examples")
            print("3. Try practice exercise")
            print("4. Return to main menu")
            
            choice = input("Your choice: ")
            
            if choice == '1':
                question = input("What's your question? ")
                response = self.chains['explain'].run({
                    'topic': topic,
                    'human_input': question,
                    'learning_style': self.user_profile['learning_style'],
                    'difficulty_level': self.user_profile['difficulty_level'],
                    'interests': self.user_profile['interests'],
                    'strengths': self.user_profile['strengths']
                })
                print(response)
                
            elif choice == '2':
                examples = self.chains['example'].run({
                    'topic': topic,
                    'human_input': topic,
                    'learning_style': self.user_profile['learning_style'],
                    'difficulty_level': self.user_profile['difficulty_level']
                })
                print("\nPractical Examples:")
                print(examples)
                
            elif choice == '3':
                score = self._conduct_quiz(topic)
                self._update_learning_profile({
                    'topic': topic,
                    'score': score,
                    'timestamp': datetime.now().isoformat(),
                    'difficulty_level': self.user_profile['difficulty_level']
                })
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")
   
        
    def handle_new_topic(self):
        """Guide user through new topic learning with initial questions"""
        self.topic = self.get_topic()
        if not self.topic:
            return
            
        if not self.student_profile:
            self.create_student_profile()
            
        profile_text = self.format_profile_for_prompt()
        
        # Initial explanation
        explanation = self.chains['explain'].run(
            topic=self.topic,
            human_input="Initial explanation request",
            profile=profile_text
        )
        
        print(f"\nLet's learn about {self.topic}!")
        print(explanation)
        
        # Prompt to ask a question
        while True:
            print("\nDo you have any questions about this topic? (Enter 'no' to continue)")
            question = input("Your question: ").strip()
            
            if question.lower() == 'no':
                break
            
            # Generate response to question
            response = self.chains['explain'].run(
                topic=self.topic,
                human_input=question,
                profile=profile_text
            )
            print("\nAnswer:", response)
            
            # Check understanding
            while True:
                understanding = input("\nDoes this answer help you understand better? (yes/no): ").lower()
                if understanding in ['yes', 'no']:
                    break
                print("Please answer with 'yes' or 'no'")
            
            if understanding == 'no':
                print("\nLet me try to explain it differently...")
                alternative_response = self.chains['explain'].run(
                    topic=self.topic,
                    human_input=f"Please explain {question} differently",
                    profile=profile_text
                )
                print(alternative_response)
        
        # Start interactive learning loop
        print("\nGreat! Let's continue with more learning activities.")
        self.interactive_learning_loop(self.topic)
        
        
    def _update_learning_profile(self, topic: str, quiz_success: bool):
        """Update student profile based on learning progress"""
        if not self.student_profile:
            return
            
        # Simple difficulty adjustment based on quiz performance
        if quiz_success:
            print("\nGreat progress! Would you like to try a harder difficulty? (yes/no)")
            if input().lower() == 'yes':
                current_level = self.student_profile.difficulty_level
                levels = list(DifficultyLevel)
                current_index = levels.index(current_level)
                if current_index < len(levels) - 1:
                    self.student_profile.difficulty_level = levels[current_index + 1]
                    print(f"Difficulty increased to {self.student_profile.difficulty_level.value}")
        else:
            print("\nWould you like to try an easier difficulty? (yes/no)")
            if input().lower() == 'yes':
                current_level = self.student_profile.difficulty_level
                levels = list(DifficultyLevel)
                current_index = levels.index(current_level)
                if current_index > 0:
                    self.student_profile.difficulty_level = levels[current_index - 1]
                    print(f"Difficulty adjusted to {self.student_profile.difficulty_level.value}")

    def ask_questions(self):
        """Handle question-asking functionality"""
        if not self.student_profile or not self.topic:
            print("Error: Student profile or topic not set")
            return
            
        profile_text = self.format_profile_for_prompt()
        
        while True:
            print("\nWhat would you like to ask about", self.topic + "?")
            print("(Type 'back' to return to previous menu)")
            
            question = input("Your question: ").strip()
            
            if question.lower() == 'back':
                break
            
            # Generate response to question
            response = self.chains['explain'].run(
                topic=self.topic,
                human_input=question,
                profile=profile_text
            )
            print("\nAnswer:", response)
            
            # Check understanding
            while True:
                understanding = input("\nDoes this answer help you understand better? (yes/no): ").lower()
                if understanding in ['yes', 'no']:
                    break
                print("Please answer with 'yes' or 'no'")
            
            if understanding == 'no':
                print("\nLet me try to explain it differently...")
                alternative_response = self.chains['explain'].run(
                    topic=self.topic,
                    human_input=f"Please explain {question} differently",
                    profile=profile_text
                )
                print(alternative_response)

    def start_lesson(self):
        """Modified start_lesson with improved flow control"""
        if not self.student_profile:
            self.create_student_profile()
        
        while True:
            result = None
            if not self.topic:
                print("\nWhat would you like to do?")
                print("1. Start new topic")
                print("2. Update your profile")
                print("3. End session")
                
                choice = input("Enter your choice (1/2/3): ")
                
                if choice == '1':
                    self.handle_new_topic()
                elif choice == '2':
                    self.create_student_profile()
                elif choice == '3':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
            else:
                print("\nWhat would you like to do next?")
                print("1. Ask a question")
                print("2. Continue with current topic")
                print("3. Start new topic")
                print("4. End session")
                
                choice = input("Enter your choice (1/2/3/4): ")
                
                if choice == '1':
                    self.ask_questions()
                elif choice == '2':
                    self.interactive_learning_loop(self.topic)
                elif choice == '3':
                    self.topic = None  # Reset topic
                    continue
                elif choice == '4':
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")

    def _parse_quiz_response(self, response: str) -> dict:
        """Parse raw quiz response into structured format"""
        quiz_data = {
            "question": "",
            "options": {},
            "correct_answer": "",
            "explanation": "",
            "hint": "",
            "difficulty": 1  # Default difficulty level
        }
        
        current_section = None
        option_key = None
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Section headers
            if line.lower().startswith("question:"):
                current_section = "question"
                quiz_data["question"] = line[len("question:"):].strip()
            elif line.lower().startswith("options:"):
                current_section = "options"
            elif line.lower().startswith("correct_answer:"):
                current_section = "correct_answer"
                quiz_data["correct_answer"] = line[len("correct_answer:"):].strip().lower()
            elif line.lower().startswith("explanation:"):
                current_section = "explanation"
                quiz_data["explanation"] = line[len("explanation:"):].strip()
            elif line.lower().startswith("hint:"):
                current_section = "hint"
                quiz_data["hint"] = line[len("hint:"):].strip()
            elif line.lower().startswith("difficulty:"):
                quiz_data["difficulty"] = int(line[len("difficulty:"):].strip())
                
            # Handle content
            elif current_section == "question":
                quiz_data["question"] += " " + line
            elif current_section == "options" and ')' in line:
                key, text = line.split(')', 1)
                option_key = key.strip().lower()
                quiz_data["options"][option_key] = text.strip()
            elif current_section == "options" and option_key:
                quiz_data["options"][option_key] += " " + line
            elif current_section == "explanation":
                quiz_data["explanation"] += " " + line
            elif current_section == "hint":
                quiz_data["hint"] += " " + line
                
        # Cleanup and validation
        quiz_data["question"] = quiz_data["question"].strip()
        quiz_data["explanation"] = quiz_data["explanation"].strip()
        quiz_data["hint"] = quiz_data["hint"].strip()
        
        # Ensure all required fields are present
        if not all([quiz_data["question"], quiz_data["options"], quiz_data["correct_answer"]]):
            print("❌ Invalid quiz format: Missing required fields.")
            return None
            
        # Ensure correct_answer is one of the options
        if quiz_data["correct_answer"] not in quiz_data["options"]:
            print(f"❌ Invalid correct_answer: {quiz_data['correct_answer']} not in options.")
            return None
            
        return quiz_data


    def conduct_personalized_quiz(self):
        """Conduct a quiz tailored to the student's profile"""
        if not self.student_profile or not self.topic:
            print("Error: Student profile or topic not set")
            return False
            
        # Format profile for prompts
        profile_text = self.format_profile_for_prompt()
        
        # Get chat history
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        # Create and run quiz instruction
        quiz_instruction = self.create_quiz_instruction(
            self.topic,
            profile_text,
            chat_history
        )
        quiz_response = self.personalized_quiz_chain.run(instruction=quiz_instruction)
        
        # Changed from parse_quiz_response to _parse_quiz_response
        quiz_data = self._parse_quiz_response(quiz_response)
        
        # Added error handling for invalid quiz data
        if not quiz_data:
            print("Sorry, there was an error generating the quiz. Let's try again.")
            return False

        print("\nQuiz Question (Difficulty: {}):"
            .format(self.student_profile.difficulty_level.value))
        print(quiz_data['question'])
        print("\nOptions:")
        for option, text in quiz_data['options'].items():
            print(f"{option}) {text}")
        
        # First attempt
        while True:
            answer = input("\nYour answer (a/b/c): ").lower().strip()
            if answer in ['a', 'b', 'c']:
                break
            print("Please enter 'a', 'b', or 'c'")
        
        if answer == quiz_data['correct_answer']:
            print("\nCorrect! Well done!")
            print("Explanation:", quiz_data['explanation'])
            return True
        else:
            while True:
                want_hint = input("\nThat's not quite right. Would you like a hint? (yes/no): ").lower().strip()
                if want_hint in ['yes', 'no']:
                    break
                print("Please answer with 'yes' or 'no'")
            
            if want_hint == 'yes':
                print("\nHint:", quiz_data['hint'])
            
            # Second attempt
            while True:
                answer = input("\nTry one more time! Your answer (a/b/c): ").lower().strip()
                if answer in ['a', 'b', 'c']:
                    break
                print("Please enter 'a', 'b', or 'c'")
            
            if answer == quiz_data['correct_answer']:
                print("\nCorrect! You got it on the second try!")
                print("Explanation:", quiz_data['explanation'])
                return True
            else:
                print("\nIncorrect. The correct answer was:", quiz_data['correct_answer'])
                print("Explanation:", quiz_data['explanation'])
                return False
        
        
    def create_student_profile(self):
        """Interactive method to create a student profile"""
        print("\nLet's create your learning profile!")
        
        # Get grade level
        while True:
            try:
                grade = int(input("What grade are you in? (1-12): "))
                if 1 <= grade <= 12:
                    break
                print("Please enter a grade between 1 and 12.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get difficulty level
        print("\nSelect your preferred difficulty level:")
        for i, level in enumerate(DifficultyLevel, 1):
            print(f"{i}. {level.value}")
        while True:
            try:
                diff_choice = int(input("Enter the number of your choice: "))
                if 1 <= diff_choice <= len(DifficultyLevel):
                    difficulty = list(DifficultyLevel)[diff_choice-1]
                    break
                print("Please enter a valid number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get learning style
        print("\nWhat's your preferred learning style?")
        for i, style in enumerate(LearningStyle, 1):
            print(f"{i}. {style.value}")
        while True:
            try:
                style_choice = int(input("Enter the number of your choice: "))
                if 1 <= style_choice <= len(LearningStyle):
                    learning_style = list(LearningStyle)[style_choice-1]
                    break
                print("Please enter a valid number.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get preferred subjects
        print("\nWhat subjects interest you most? (comma-separated list)")
        subjects = [subj.strip() for subj in input().split(",")]
        
        # Get attention span
        while True:
            try:
                attention = int(input("\nHow many minutes can you typically focus? (5-60): "))
                if 5 <= attention <= 60:
                    break
                print("Please enter a duration between 5 and 60 minutes.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get prior knowledge
        prior = input("\nWhat do you already know about this subject? (Press Enter to skip): ").strip()
        
        self.student_profile = StudentProfile(
            grade_level=grade,
            difficulty_level=difficulty,
            learning_style=learning_style,
            preferred_subjects=subjects,
            attention_span=attention,
            prior_knowledge=prior if prior else None
        )
        
        print("\nProfile created successfully!")
        return self.student_profile

   
# Example usage
if __name__ == "__main__":
    agent = PersonalizedTeachingAgent()
    agent.start_lesson()