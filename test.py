import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import os
import re

# Set page title
st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

# Initialize session state variables
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
if 'topic' not in st.session_state:
    st.session_state.topic = None
if 'topic_validated' not in st.session_state:
    st.session_state.topic_validated = False
if 'explanation_shown' not in st.session_state:
    st.session_state.explanation_shown = False
if 'explanation_text' not in st.session_state:
    st.session_state.explanation_text = ""
if 'examples_shown' not in st.session_state:
    st.session_state.examples_shown = False
if 'examples_text' not in st.session_state:
    st.session_state.examples_text = ""
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'quiz_answered' not in st.session_state:
    st.session_state.quiz_answered = False
if 'quiz_correct' not in st.session_state:
    st.session_state.quiz_correct = False
if 'hint_shown' not in st.session_state:
    st.session_state.hint_shown = False
if 'second_attempt' not in st.session_state:
    st.session_state.second_attempt = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'name': 'Guest',
        'age': 25,
        'interests': [],
        'learning_style': 'visual',
        'setup_complete': False
    }

class TeachingAgent:
    def __init__(self):
        # Initialize the LLM
        api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=api_key)
        
        # Define prompts
        self.topic_validation_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Evaluate if {topic} is a valid educational topic that can be taught.
            If valid, respond with "VALID: [cleaned topic]".
            If invalid, respond with "INVALID: [reason]".
            """
        )
        
        self.explain_prompt = PromptTemplate(
            input_variables=["topic", "user_name", "user_age", "learning_style"],
            template="""
            Explain {topic} for {user_name} (age {user_age}) using {learning_style} learning style.
            Use age-appropriate language and relevant analogies.
            Keep it concise and engaging. No examples yet.
            """
        )
        
        self.example_prompt = PromptTemplate(
            input_variables=["topic", "chat_history", "user_interests"],
            template="""
            Create 2-3 examples about {topic} related to these interests: {user_interests}.
            Make them practical and relatable. Previous conversation: {chat_history}
            """
        )
        
        self.quiz_prompt = PromptTemplate(
            input_variables=["topic", "chat_history", "user_name", "user_interests"],
            template="""
            Create 3 personalized multiple choice questions about {topic} for {user_name}
            incorporating these interests: {user_interests}. Use this exact format:
            
            QUESTION1: [personalized question]
            OPTIONS:
            a) [option 1]
            b) [option 2]
            c) [option 3]
            CORRECT_ANSWER: [letter]
            EXPLANATION: [personalized explanation]
            HINT: [personalized hint]
            
            [Repeat for QUESTIONS 2-3 with increasing difficulty]
            Previous conversation: {chat_history}
            """
        )
        
        self.question_prompt = PromptTemplate(
            input_variables=["topic", "question", "chat_history", "user_name", "learning_style"],
            template="""
            Answer {user_name}'s question about {topic} using {learning_style} learning style.
            Be concise and helpful. If unrelated, gently guide back.
            Question: {question}
            Previous conversation: {chat_history}
            """
        )
        
        # Initialize chains
        self.topic_validation_chain = LLMChain(llm=self.llm, prompt=self.topic_validation_prompt)
        self.explain_chain = LLMChain(llm=self.llm, prompt=self.explain_prompt)
        self.example_chain = LLMChain(llm=self.llm, prompt=self.example_prompt)
        self.quiz_chain = LLMChain(llm=self.llm, prompt=self.quiz_prompt)
        self.question_chain = LLMChain(llm=self.llm, prompt=self.question_prompt)
    
    def validate_topic(self, topic):
        response = self.topic_validation_chain.run(topic=topic)
        return (True, response[6:].strip()) if response.startswith("VALID:") else (False, response[8:].strip())

    def parse_quiz_response(self, response):
        """Parse the quiz response from LLM into a list of question dictionaries"""
        quiz_data = []
        questions = re.split(r'(?i)(?=QUESTION\d+:)', response)  # Case-insensitive split
        
        for question in questions:
            if not question.strip():
                continue
                
            question_data = {
                'question': '',
                'options': {},
                'correct_answer': '',
                'explanation': '',
                'hint': ''
            }
            
            try:
                # Extract question text (more flexible matching)
                question_match = re.search(r'(?i)QUESTION\d+:\s*(.*?)(?=OPTIONS:|$)', question, re.DOTALL)
                if question_match:
                    question_data['question'] = question_match.group(1).strip()
                
                # Extract options with more flexible parsing
                options_match = re.search(r'(?i)OPTIONS:(.*?)(?=CORRECT_ANSWER|EXPLANATION|$)', question, re.DOTALL)
                if options_match:
                    options_text = options_match.group(1)
                    for line in options_text.split('\n'):
                        line = line.strip()
                        option_match = re.match(r'(?i)^([a-c])[).]\s*(.*)$', line)
                        if option_match:
                            option = option_match.group(1).lower()
                            text = option_match.group(2).strip()
                            question_data['options'][option] = text
                
                # Extract correct answer with flexible formatting
                correct_match = re.search(r'(?i)CORRECT_ANSWER:\s*([a-c])', question)
                if correct_match:
                    question_data['correct_answer'] = correct_match.group(1).lower()
                
                # Extract explanation with flexible matching
                explanation_match = re.search(r'(?i)EXPLANATION:\s*(.*?)(?=HINT:|$)', question, re.DOTALL)
                if explanation_match:
                    question_data['explanation'] = explanation_match.group(1).strip()
                
                # Extract hint with flexible matching
                hint_match = re.search(r'(?i)HINT:\s*(.*)', question, re.DOTALL)
                if hint_match:
                    question_data['hint'] = hint_match.group(1).strip()
                
                # Validate all required fields
                if (question_data['question'] and 
                    len(question_data['options']) >= 2 and  # Require at least 2 options
                    question_data['correct_answer'] in ['a', 'b', 'c'] and
                    question_data['explanation']):
                    quiz_data.append(question_data)
                    
            except (AttributeError, IndexError) as e:
                continue  # Skip malformed questions
                
        return quiz_data if quiz_data else None
def collect_user_profile():
    st.subheader("Let's personalize your experience!")
    with st.form("user_profile"):
        name = st.text_input("What's your name?", value=st.session_state.user_profile['name'])
        age = st.number_input("Your age", min_value=5, max_value=100, 
                            value=st.session_state.user_profile['age'])
        interests = st.multiselect(
            "Your interests (select up to 3)",
            options=['Sports', 'Tech', 'Art', 'Music', 'Science', 'History', 'Cooking'],
            default=st.session_state.user_profile['interests'],
            max_selections=3
        )
        learning_style = st.radio(
            "Your learning style",
            options=['Visual', 'Verbal', 'Hands-on', 'Theoretical'],
            index=['visual', 'verbal', 'hands-on', 'theoretical'].index(
                st.session_state.user_profile['learning_style'])
        )
        if st.form_submit_button("Save Preferences"):
            st.session_state.user_profile.update({
                'name': name,
                'age': age,
                'interests': interests,
                'learning_style': learning_style.lower(),
                'setup_complete': True
            })
            st.rerun()

# Initialize the teaching agent
agent = TeachingAgent()

# UI Elements
st.title("AI Teaching Assistant")
st.markdown("Your personalized learning companion")

# Topic Input
if not st.session_state.topic_validated:
    st.subheader("What would you like to learn about?")
    topic_input = st.text_input("Enter a topic:", key="topic_input")
    
    if st.button("Start Learning"):
        if topic_input:
            is_valid, message = agent.validate_topic(topic_input)
            if is_valid:
                st.session_state.topic = message
                st.session_state.topic_validated = True
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
                st.rerun()
            else:
                st.error(f"Invalid topic: {message}")
        else:
            st.warning("Please enter a topic")

# Learning Interface
if st.session_state.topic_validated:
    if not st.session_state.user_profile['setup_complete']:
        collect_user_profile()
    else:
        st.subheader(f"Hi {st.session_state.user_profile['name']}! Learning about: {st.session_state.topic}")
        
        # Step 1: Personalized Explanation
        if not st.session_state.explanation_shown:
            with st.spinner("Crafting your personalized explanation..."):
                explanation_response = agent.explain_chain.run(
                    topic=st.session_state.topic,
                    user_name=st.session_state.user_profile['name'],
                    user_age=st.session_state.user_profile['age'],
                    learning_style=st.session_state.user_profile['learning_style']
                )
                st.session_state.memory.save_context(
                    {"input": f"Explain {st.session_state.topic}"},
                    {"output": explanation_response}
                )
                st.session_state.explanation_text = explanation_response
                st.session_state.explanation_shown = True
        
        st.markdown("### Personalized Explanation")
        st.write(st.session_state.explanation_text)

        # Question Asking during Explanation
        user_question = st.text_input("Any questions about the explanation?", key="question_input_explanation")
        if user_question:
            with st.spinner("Tailoring your answer..."):
                question_response = agent.question_chain.run(
                    topic=st.session_state.topic,
                    question=user_question,
                    chat_history=st.session_state.memory.buffer,
                    user_name=st.session_state.user_profile['name'],
                    learning_style=st.session_state.user_profile['learning_style']
                )
                st.session_state.memory.save_context(
                    {"input": user_question},
                    {"output": question_response}
                )
                st.markdown("### Custom Answer")
                st.write(question_response)

        # Step 2: Understanding Check
        understanding = st.radio(
            "How's this explanation working for you?",
            ["I'm following along!", "I need a different approach"],
            key="understanding"
        )
        
        if understanding == "I need a different approach":
            if st.button("Try Alternative Explanation"):
                with st.spinner("Adapting to your learning style..."):
                    explanation = agent.explain_chain.run(
                        topic=st.session_state.topic,
                        user_name=st.session_state.user_profile['name'],
                        user_age=st.session_state.user_profile['age'],
                        learning_style=st.session_state.user_profile['learning_style']
                    )
                    st.session_state.memory.save_context(
                        {"input": f"Alternative explanation of {st.session_state.topic}"},
                        {"output": explanation}
                    )
                    st.markdown("### Alternative Explanation")
                    st.write(explanation)
        
        # Step 3: Personalized Examples
        if understanding == "I'm following along!" and not st.session_state.examples_shown:
            if st.button("Show Me Examples"):
                with st.spinner("Generating personalized examples..."):
                    examples = agent.example_chain.run(
                        topic=st.session_state.topic,
                        chat_history=st.session_state.memory.buffer,
                        user_interests=st.session_state.user_profile['interests']
                    )
                    st.session_state.memory.save_context(
                        {"input": f"Examples for {st.session_state.topic}"},
                        {"output": examples}
                    )
                    st.session_state.examples_text = examples
                    st.session_state.examples_shown = True
        
        if st.session_state.examples_shown:
            st.markdown("### Customized Examples")
            st.write(st.session_state.examples_text)
            
            # Question Asking during Examples
            user_question = st.text_input("Questions about the examples?", key="question_input_examples")
            if user_question:
                with st.spinner("Preparing your answer..."):
                    question_response = agent.question_chain.run(
                        topic=st.session_state.topic,
                        question=user_question,
                        chat_history=st.session_state.memory.buffer,
                        user_name=st.session_state.user_profile['name'],
                        learning_style=st.session_state.user_profile['learning_style']
                    )
                    st.session_state.memory.save_context(
                        {"input": user_question},
                        {"output": question_response}
                    )
                    st.markdown("### Tailored Response")
                    st.write(question_response)
            
            if st.button("More Personalized Examples"):
                with st.spinner("Generating additional examples..."):
                    more_examples = agent.example_chain.run(
                        topic=st.session_state.topic,
                        chat_history=st.session_state.memory.buffer,
                        user_interests=st.session_state.user_profile['interests']
                    )
                    st.session_state.memory.save_context(
                        {"input": f"More examples for {st.session_state.topic}"},
                        {"output": more_examples}
                    )
                    st.markdown("### Additional Examples")
                    st.write(more_examples)
        
        # Step 4: Personalized Quiz
        if st.session_state.examples_shown and not st.session_state.quiz_data:
            if st.button("Take Personalized Quiz"):
                with st.spinner("Creating your custom quiz..."):
                    quiz_response = agent.quiz_chain.run(
                        topic=st.session_state.topic,
                        chat_history=st.session_state.memory.buffer,
                        user_name=st.session_state.user_profile['name'],
                        user_interests=st.session_state.user_profile['interests']
                    )
                    st.session_state.quiz_data = agent.parse_quiz_response(quiz_response)
                    st.session_state.memory.save_context(
                        {"input": f"Quiz about {st.session_state.topic}"},
                        {"output": quiz_response}
                    )
        
        if st.session_state.quiz_data and not st.session_state.quiz_answered:
            if 'current_question_index' not in st.session_state:
                st.session_state.current_question_index = 0
                st.session_state.quiz_results = []
                st.session_state.show_results = False
            
            current_question = st.session_state.quiz_data[st.session_state.current_question_index]
            
            st.markdown(f"### {st.session_state.user_profile['name']}'s Quiz")
            st.markdown(f"**Question {st.session_state.current_question_index + 1}:** {current_question['question']}")
            
            options = list(current_question['options'].items())
            quiz_answer = st.radio(
                "Your answer:",
                options=[(o[0], o[1]) for o in options],
                format_func=lambda x: f"{x[0]}) {x[1]}",
                key=f"quiz_answer_{st.session_state.current_question_index}"
            )
            
            if st.button("Submit Answer"):
                is_correct = quiz_answer[0] == current_question['correct_answer']
                st.session_state.quiz_results.append({
                    'question': current_question['question'],
                    'correct': is_correct,
                    'user_answer': quiz_answer[0],
                    'correct_answer': current_question['correct_answer'],
                    'explanation': current_question['explanation']
                })
                
                if is_correct:
                    st.success(f"Great job {st.session_state.user_profile['name']}! 🎉")
                else:
                    st.error("Let's try that again")
                    if not st.session_state.hint_shown:
                        st.info(f"**Personalized Hint:** {current_question['hint']}")
                        st.session_state.hint_shown = True
                        st.session_state.second_attempt = True
                    else:
                        st.markdown(f"**Explanation:** {current_question['explanation']}")
                        st.session_state.current_question_index += 1
                        st.session_state.hint_shown = False
                
                if st.session_state.current_question_index >= len(st.session_state.quiz_data):
                    st.session_state.quiz_answered = True
                    st.session_state.show_results = True
            
            if st.session_state.show_results:
                st.markdown("### Quiz Results")
                correct_count = sum(r['correct'] for r in st.session_state.quiz_results)
                st.markdown(f"**Way to go {st.session_state.user_profile['name']}! Score: {correct_count}/{len(st.session_state.quiz_data)}**")
                for i, result in enumerate(st.session_state.quiz_results):
                    st.markdown(f"**Q{i+1}:** {result['question']}")
                    st.markdown(f"Your answer: {result['user_answer']})")
                    st.markdown(f"Correct answer: {result['correct_answer']})")
                    st.markdown(f"**Insight:** {result['explanation']}")
        
        # Navigation
        if st.session_state.examples_shown:
            st.markdown("### Next Steps")
            if st.button("🔄 Start New Topic"):
                for key in ['topic', 'topic_validated', 'explanation_shown', 'examples_shown',
                           'quiz_data', 'quiz_answered', 'hint_shown', 'second_attempt']:
                    st.session_state[key] = False if isinstance(st.session_state[key], bool) else None
                st.rerun()
            if st.button("✏️ Edit Profile"):
                st.session_state.user_profile['setup_complete'] = False
                st.rerun()