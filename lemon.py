from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from datetime import datetime
from typing import List

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

class PersonalizedTutor:
    def __init__(self, api_key: str):
        self.chat_model = ChatOpenAI(
            model="deepseek/deepseek-chat",
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=""
        )

    def generate_response(self, question: str) -> str:
        """Generate a response using DeepSeek model via OpenRouter"""
        response = self.chat_model.invoke(question)
        return response.content

# Example usage
if __name__ == "__main__":
    tutor = PersonalizedTutor(api_key="your-api-key-here")
    question = "Can you explain quadratic equations?"
    response = tutor.generate_response(question)
    print(f"AI Response: {response}")
