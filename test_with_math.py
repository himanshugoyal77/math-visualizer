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
        
        # Simplified prompts with single input
        self.profile_prompt = PromptTemplate(
            input_variables=["instruction"],
            template="{instruction}"
        )
        
        self.personalized_example_prompt = PromptTemplate(
            input_variables=["instruction"],
            template="{instruction}"
        )
        
        self.personalized_quiz_prompt = PromptTemplate(
            input_variables=["instruction"],
            template="{instruction}"
        )
        
        # Initialize chains
        self.profile_chain = LLMChain(
            llm=self.llm,
            prompt=self.profile_prompt,
            memory=self.memory
        )
        
        self.personalized_example_chain = LLMChain(
            llm=self.llm,
            prompt=self.personalized_example_prompt,
            memory=self.memory
        )
        
        self.personalized_quiz_chain = LLMChain(
            llm=self.llm,
            prompt=self.personalized_quiz_prompt,
            memory=self.memory
        )

    def format_profile_for_prompt(self):
        """Format student profile as a string for prompt templates"""
        if not self.student_profile:
            return None
        
        return f"""
- Grade Level: {self.student_profile.grade_level}
- Difficulty Level: {self.student_profile.difficulty_level.value}
- Learning Style: {self.student_profile.learning_style.value}
- Preferred Subjects: {', '.join(self.student_profile.preferred_subjects)}
- Attention Span: {self.student_profile.attention_span} minutes
- Prior Knowledge: {self.student_profile.prior_knowledge or 'None specified'}
        """.strip()

    def create_explanation_instruction(self, topic, profile):
        return f"""
Adapt the explanation of {topic} for a student with the following profile:
{profile}

Tailor your explanation to match their learning style and keep within their attention span.
Use grade-appropriate vocabulary and examples.
        """.strip()

    def create_example_instruction(self, topic, profile, chat_history):
        return f"""
Create examples of {topic} for a student with the following profile:
{profile}

Make the examples relatable and appropriate for their grade level and learning style.
Previous conversation: {chat_history}
        """.strip()

    def create_quiz_instruction(self, topic, profile, chat_history):
        return f"""
Create a multiple choice question about {topic} for a student with the following profile:
{profile}

Structure your response exactly as follows:
QUESTION: (grade-appropriate question)
OPTIONS:
a) (first option)
b) (second option)
c) (third option)
CORRECT_ANSWER: (letter)
EXPLANATION: (grade-appropriate explanation)
HINT: (helpful hint matching their learning style)

Previous conversation: {chat_history}
        """.strip()

    def start_lesson(self):
        """Override start_lesson to include personalization"""
        if not self.student_profile:
            self.create_student_profile()
        
        while True:
            self.topic = self.get_topic()
            if not self.topic:
                print("Goodbye!")
                break
            
            # Reset memory for new topic
            self.memory = ConversationBufferMemory(memory_key="chat_history")
            
            # Format profile for prompts
            profile_text = self.format_profile_for_prompt()
            
            # Create and run explanation instruction
            explanation_instruction = self.create_explanation_instruction(
                self.topic, 
                profile_text
            )
            explanation = self.profile_chain.run(instruction=explanation_instruction)
            
            print("\nLet me explain", self.topic)
            print(explanation)
            
            # Check understanding
            while True:
                understanding = input("\nDo you understand this explanation? (yes/no): ").lower()
                
                if understanding == 'no':
                    print("\nLet me explain it differently, keeping in mind your learning style...")
                    explanation = self.profile_chain.run(instruction=explanation_instruction)
                    print(explanation)
                elif understanding == 'yes':
                    break
                else:
                    print("Please answer with 'yes' or 'no'")
            
            # Create and run example instruction
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            example_instruction = self.create_example_instruction(
                self.topic,
                profile_text,
                chat_history
            )
            
            print("\nHere are some examples tailored to your interests and learning style...")
            examples = self.personalized_example_chain.run(instruction=example_instruction)
            print(examples)
            
            while True:
                print("\nWhat would you like to do next?")
                print("1. See more personalized examples")
                print("2. Take a quiz at your level")
                print("3. Start new topic")
                print("4. Update your profile")
                print("5. End session")
                
                choice = input("Enter your choice (1/2/3/4/5): ")
                
                if choice == '1':
                    print("\nHere are more examples aligned with your learning style...")
                    examples = self.personalized_example_chain.run(instruction=example_instruction)
                    print(examples)
                elif choice == '2':
                    print("\nLet's test your understanding with a quiz at your level!")
                    quiz_result = self.conduct_personalized_quiz()
                    
                    print("\nWould you like to:")
                    print("1. Continue with current topic")
                    print("2. Start new topic")
                    print("3. End session")
                    
                    end_choice = input("Enter your choice (1/2/3): ")
                    if end_choice == '2':
                        break
                    elif end_choice == '3':
                        return
                elif choice == '3':
                    break
                elif choice == '4':
                    self.create_student_profile()
                    profile_text = self.format_profile_for_prompt()
                elif choice == '5':
                    print("Goodbye!")
                    return
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

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
        
        quiz_data = self.parse_quiz_response(quiz_response)
        
        print("\nQuiz Question (Difficulty: {}):"
              .format(self.student_profile.difficulty_level.value))
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