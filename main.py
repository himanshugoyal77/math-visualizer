from typing import TypedDict, Sequence, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from enum import Enum
from datetime import datetime

# Define the student profile structure
class StudentProfile(BaseModel):
    name: str
    grade_level: int
    subjects: List[str]
    learning_style: str  # "visual", "auditory", "kinesthetic"
    comprehension_level: int  # 1-5 scale
    previous_topics: List[str]
    struggles: List[str]
    strengths: List[str]
    
# Define interaction history
class Interaction(BaseModel):
    timestamp: datetime
    topic: str
    comprehension_score: int  # 1-5 scale
    engagement_level: int  # 1-5 scale
    notes: str

# State management for the tutor
class TutorState(TypedDict):
    messages: Sequence[BaseMessage]
    student_profile: StudentProfile
    interaction_history: List[Interaction]
    current_complexity: int  # 1-5 scale

class PersonalizedTutor:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="deepseek/deepseek-chat",
            base_url="https://openrouter.ai/api/v1", 
            api_key=""
        )
        
        # Create the base system prompt
        self.base_prompt = """
        You are an expert AI tutor specializing in personalized education. 
        
        Student Profile:
        Name: {name}
        Grade Level: {grade}
        Learning Style: {learning_style}
        Current Comprehension Level: {comprehension}
        Strengths: {strengths}
        Struggles: {struggles}
        
        Previous Topics Covered: {previous_topics}
        
        Your task is to:
        1. Explain concepts at the appropriate complexity level ({complexity}/5)
        2. Use examples and explanations that match the student's learning style
        3. Build upon their strengths and previous knowledge
        4. Provide extra support in areas they struggle with
        5. Adjust your language and complexity based on their responses
        
        Current Topic: {topic}
        
        Remember to:
        - Use {learning_style} learning style techniques
        - Keep explanations at complexity level {complexity}
        - Reference previous topics when relevant
        - Check for understanding frequently
        - Provide encouraging feedback
        
        Previous interaction context:
        {interaction_history}
        
        Respond to the following question:
        """
        
    def adjust_complexity(self, current_complexity: int, comprehension_score: int) -> int:
        """Adjust complexity based on student's comprehension"""
        if comprehension_score <= 2:
            return max(1, current_complexity - 1)
        elif comprehension_score >= 4:
            return min(5, current_complexity + 1)
        return current_complexity

    def create_interaction(self, topic: str, comprehension_score: int, 
                         engagement_level: int, notes: str) -> Interaction:
        """Record a new interaction with the student"""
        return Interaction(
            timestamp=datetime.now(),
            topic=topic,
            comprehension_score=comprehension_score,
            engagement_level=engagement_level,
            notes=notes
        )

    def format_interaction_history(self, history: List[Interaction]) -> str:
        """Format recent interaction history for context"""
        recent_history = history[-3:]  # Get last 3 interactions
        formatted = []
        for interaction in recent_history:
            formatted.append(
                f"Topic: {interaction.topic}\n"
                f"Comprehension: {interaction.comprehension_score}/5\n"
                f"Notes: {interaction.notes}\n"
            )
        return "\n".join(formatted)

    def generate_response(self, 
                        question: str,
                        student_profile: StudentProfile,
                        interaction_history: List[Interaction],
                        current_complexity: int,
                        topic: str) -> str:
        """Generate a personalized response to the student's question"""

        # Format the prompt with student information
        formatted_prompt = self.base_prompt.format(
            name=student_profile.name,
            grade=student_profile.grade_level,
            learning_style=student_profile.learning_style,
            comprehension=student_profile.comprehension_level,
            strengths=", ".join(student_profile.strengths),
            struggles=", ".join(student_profile.struggles),
            previous_topics=", ".join(student_profile.previous_topics),
            complexity=current_complexity,
            topic=topic,
            interaction_history=self.format_interaction_history(interaction_history)
        )

        # Create the full prompt
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(formatted_prompt),
            HumanMessage(content=question)
        ])

        # **Fix: Convert the prompt into a message list before invoking**
        formatted_prompt_messages = prompt_template.format_messages()

        # Generate the response
        response = self.llm.invoke(formatted_prompt_messages)
        return response.content


    def update_student_profile(self, 
                             profile: StudentProfile,
                             new_topic: str,
                             comprehension_score: int) -> StudentProfile:
        """Update student profile based on new interaction"""
        profile.previous_topics.append(new_topic)
        
        # Update comprehension level (rolling average)
        profile.comprehension_level = int((profile.comprehension_level + comprehension_score) / 2)
        
        return profile

# Example usage
def create_sample_student():
    return StudentProfile(
        name="Alex",
        grade_level=8,
        subjects=["Math", "Science"],
        learning_style="visual",
        comprehension_level=3,
        previous_topics=["Algebra basics", "Linear equations"],
        struggles=["Word problems", "Fractions"],
        strengths=["Geometry", "Pattern recognition"]
    )
    
    # Initialize interaction history
interaction_history = []

def main(question):
    # Initialize the tutor
    tutor = PersonalizedTutor(api_key="your-api-key")
    
    # Create a student profile
    student = create_sample_student()
    current_complexity = 3
    
    print("initialized", interaction_history)
    print("initialized", current_complexity)
    
    
    # Example interaction
    question = question
    topic = "Quadratic equations"
    
    # Generate response
    response = tutor.generate_response(
        question=question,
        student_profile=student,
        interaction_history=interaction_history,
        current_complexity=current_complexity,
        topic=topic
    )
    
    # Record interaction
    interaction = tutor.create_interaction(
        topic=topic,
        comprehension_score=3,
        engagement_level=4,
        notes="Student showed interest in visual representations"
    )
    interaction_history.append(interaction)
    
    # Update complexity and student profile
    new_complexity = tutor.adjust_complexity(current_complexity, comprehension_score=3)
    student = tutor.update_student_profile(student, topic, comprehension_score=3)
    
    return response

if __name__ == "__main__":
    while(True):
        print("====================================")
        question = input("Enter your question: ")
        if question == "exit":
            break   
        print(main(question=question))
        print("====================================")