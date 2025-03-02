import streamlit as st
import matplotlib.pyplot as plt
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import os
from io import StringIO
import contextlib
import time
import random
import json
from datetime import datetime

class TeachingAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model='google/gemini-2.0-flash-001',
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER_KEY")
        )
        
        # Initialize prompts
        self.topic_validation_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Evaluate if {topic} is a valid educational topic that can be taught.
            If it is valid, respond with "VALID: " followed by a cleaned up version of the topic name.
            If it is not valid or too broad, respond with "INVALID: " followed by a brief explanation why.
            """
        )
        
        self.explain_prompt = PromptTemplate(
            input_variables=["topic"],
            template="""
            Provide a simple and clear explanation of {topic}. 
            Keep it concise and beginner-friendly.
            
            If a visualization (chart, diagram, or graph) would enhance understanding:
            1. Describe the visualization in one sentence
            2. Generate Python code using matplotlib to create it
            3. Make sure the code:
            - Includes 'import matplotlib.pyplot as plt'
            - Uses 'plt.figure(figsize=(10, 6))' for appropriate sizing
            - Does NOT include plt.show()
            4. Mention this is optional and can be run locally
            
            Format any code with triple backticks.
            Don't provide examples yet.
            """
        )
        
        self.example_prompt = PromptTemplate(
            input_variables=["topic", "chat_history"],
            template="""
            Based on the previous explanation of {topic}, provide 2-3 practical examples.
            Make the examples relatable to everyday situations.
            Make sure to provide different examples than before if any were given.
            Previous conversation: {chat_history}
            """
        )
        
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
            """
        )
        
        # Initialize chains
        self.topic_validation_chain = LLMChain(
            llm=self.llm,
            prompt=self.topic_validation_prompt
        )
        
    def validate_topic(self, topic):
        """Validate if the topic is appropriate for teaching"""
        response = self.topic_validation_chain.run(topic=topic)
        if response.startswith("VALID:"):
            return True, response[6:].strip()
        return False, response[8:].strip()

    def get_explanation(self, topic):
        """Get explanation for a topic"""
        memory = ConversationBufferMemory(memory_key="chat_history")
        explain_chain = LLMChain(
            llm=self.llm,
            prompt=self.explain_prompt,
            memory=memory
        )
        return explain_chain.run(topic=topic), memory

    def get_examples(self, topic, memory):
        """Get examples for a topic"""
        example_chain = LLMChain(
            llm=self.llm,
            prompt=self.example_prompt,
            memory=memory
        )
        return example_chain.run(topic=topic)

    def get_quiz(self, topic, memory):
        """Get quiz for a topic"""
        quiz_chain = LLMChain(
            llm=self.llm,
            prompt=self.quiz_prompt,
            memory=memory
        )
        return quiz_chain.run(topic=topic)

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

def extract_visualization(response):
    """Extract visualization code from response"""
    if '```python' in response:
        code_part = response.split('```python')[1].split('```')[0].strip()
        return code_part
    return None

def save_feedback(feedback_data):
    """Save feedback to a JSON file"""
    try:
        with open("feedback.json", "a") as f:
            json.dump(feedback_data, f)
            f.write("\n")
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")

def load_feedback():
    """Load feedback from a JSON file"""
    try:
        with open("feedback.json", "r") as f:
            feedback_data = [json.loads(line) for line in f]
        return feedback_data
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading feedback: {str(e)}")
        return []

def main():
    st.set_page_config(page_title="AI Teaching Assistant", layout="wide")
    
    # Initialize session state variables
    for key in ['agent', 'memory', 'current_topic', 'quiz_data', 'show_hint', 
                'quiz_submitted', 'understanding_rating', 'learning_progress',
                'examples_shown', 'quiz_attempts', 'feedback_submitted']:
        if key not in st.session_state:
            st.session_state[key] = None if key not in ['quiz_attempts', 'examples_shown'] else 0
    
    # Initialize agent if not present
    if st.session_state.agent is None:
        st.session_state.agent = TeachingAgent()
    
    # Custom CSS for improved styling
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-color: #1f77b4;
        }
        .feedback-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Animated title with emoji
    st.markdown("# 🎓 AI Teaching Assistant")
    st.markdown("#### Your interactive learning companion")
    
    # Topic input with autocomplete suggestions
    common_topics = ["Python Programming", "Mathematics", "Physics", "History", "Biology", "Chemistry"]
    with st.form("topic_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            topic = st.text_input(
                "What would you like to learn about?",
                help="Type or select a topic you're interested in learning about"
            )
            st.markdown("Suggestions: " + ", ".join([f"`{t}`" for t in common_topics]))
        with col2:
            submit_topic = st.form_submit_button("Start Learning", use_container_width=True)
    
    if submit_topic and topic:
        with st.spinner("Validating topic..."):
            is_valid, message = st.session_state.agent.validate_topic(topic)
            
        if is_valid:
            st.session_state.current_topic = message
            st.session_state.learning_progress = 0
            
            # Simulate loading with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, status in enumerate([
                "Gathering knowledge...",
                "Preparing explanation...",
                "Generating examples...",
                "Creating visualizations..."
            ]):
                status_text.text(status)
                progress_bar.progress((i + 1) * 25)
                time.sleep(0.5)
            
            explanation, memory = st.session_state.agent.get_explanation(message)
            st.session_state.memory = memory
            
            # Clear loading indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display explanation in an expander
            with st.expander("📚 Explanation", expanded=True):
                st.write(explanation)
                
                # Understanding rating
                st.write("### How well did you understand this explanation?")
                understanding = st.slider(
                    "Rate your understanding",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="1 = Not at all, 5 = Perfectly clear"
                )
                
                if understanding <= 3:
                    st.info("Would you like a simpler explanation? Click 'Explain Again' below.")
                    if st.button("Explain Again"):
                        with st.spinner("Generating simpler explanation..."):
                            new_explanation, _ = st.session_state.agent.get_explanation(message)
                            st.write("### Simplified Explanation:")
                            st.write(new_explanation)
                
                # Visualization handling
                vis_code = extract_visualization(explanation)
                if vis_code:
                    st.subheader("📊 Interactive Visualization")
                    try:
                        exec(vis_code)
                        st.pyplot(plt)
                        plt.close()
                        
                        # Add interactive elements for the visualization
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
        else:
            st.error(message)
            st.info("Try being more specific or choosing from the suggested topics above.")
    
    # Only show these sections if we have a current topic
    if st.session_state.current_topic and st.session_state.memory:
        # Progress tracking
        st.session_state.learning_progress = min(100, st.session_state.learning_progress or 0 + 20)
        st.markdown("### 📈 Learning Progress")
        st.progress(st.session_state.learning_progress / 100)
        
        # Interactive tabs with badges showing completion status
        tab1, tab2, tab3 = st.tabs([
            f"📝 Examples ({st.session_state.examples_shown} shown)",
            f"❓ Quiz ({st.session_state.quiz_attempts} attempts)",
            "🔄 New Topic"
        ])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            with col1:
                example_type = st.selectbox(
                    "What type of examples would you like?",
                    ["Basic Examples", "Real-world Applications", "Advanced Cases"]
                )
            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("Show Examples", use_container_width=True):
                    with st.spinner("Generating examples..."):
                        examples = st.session_state.agent.get_examples(
                            st.session_state.current_topic,
                            st.session_state.memory
                        )
                        st.session_state.examples_shown += 1
                        st.write(examples)
                        
                        # Interactive example feedback
                        st.write("### Was this helpful?")
                        col1, col2, col3 = st.columns([1,1,2])
                        with col1:
                            if st.button("👍 Yes"):
                                st.success("Thanks for your feedback!")
                        with col2:
                            if st.button("👎 No"):
                                st.info("We'll try to provide better examples next time.")
        
        with tab2:
            if st.button("Take Quiz", use_container_width=True):
                with st.spinner("Preparing quiz..."):
                    quiz_response = st.session_state.agent.get_quiz(
                        st.session_state.current_topic,
                        st.session_state.memory
                    )
                    st.session_state.quiz_data = st.session_state.agent.parse_quiz_response(quiz_response)
                    st.session_state.quiz_submitted = False
                    st.session_state.show_hint = False
                    st.session_state.quiz_attempts += 1
            
            if st.session_state.quiz_data:
                st.subheader("Quiz Question:")
                st.info(st.session_state.quiz_data['question'])
                
                # Timer for quiz (optional)
                if not st.session_state.quiz_submitted:
                    time_limit = st.checkbox("Enable 30-second timer")
                    if time_limit:
                        timer_placeholder = st.empty()
                        for remaining in range(30, 0, -1):
                            timer_placeholder.warning(f"⏱️ Time remaining: {remaining} seconds")
                            time.sleep(1)
                            if st.session_state.quiz_submitted:
                                break
                        timer_placeholder.empty()
                        if not st.session_state.quiz_submitted:
                            st.warning("Time's up! But you can still submit your answer.")
                
                # Radio buttons for options with improved styling
                answer = st.radio(
                    "Select your answer:",
                    options=list(st.session_state.quiz_data['options'].keys()),
                    format_func=lambda x: f"{x}) {st.session_state.quiz_data['options'][x]}"
                )
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Hint button with confidence check
                    if not st.session_state.quiz_submitted:
                        confidence = st.select_slider(
                            "How confident are you?",
                            options=["Not sure", "Somewhat sure", "Very sure"]
                        )
                        if confidence == "Not sure" and st.button("💡 Show Hint"):
                            st.session_state.show_hint = True
                
                if st.session_state.show_hint:
                    st.info(f"💡 Hint: {st.session_state.quiz_data['hint']}")
                
                # Submit button with loading animation
                if not st.session_state.quiz_submitted and st.button("Submit Answer", use_container_width=True):
                    with st.spinner("Checking answer..."):
                        time.sleep(0.5)  # Brief pause for effect
                        st.session_state.quiz_submitted = True
                        if answer == st.session_state.quiz_data['correct_answer']:
                            st.balloons()
                            st.success("✅ Correct! Well done!")
                            st.session_state.learning_progress = min(100, st.session_state.learning_progress + 20)
                        else:
                            st.error("❌ That's not quite right.")
                        st.write(f"Explanation: {st.session_state.quiz_data['explanation']}")
                        
                        # Ask if they want to try another question
                        if st.button("Try Another Question"):
                            st.session_state.quiz_data = None
                            st.rerun()
        
        with tab3:
            st.write("Ready to explore a new topic?")
            if st.button("Start New Topic", use_container_width=True):
                # Save learning progress or statistics if needed
                st.session_state.current_topic = None
                st.session_state.memory = None
                st.session_state.quiz_data = None
                st.session_state.show_hint = False
                st.session_state.quiz_submitted = False
                st.session_state.learning_progress = 0
                st.session_state.examples_shown = 0
                st.session_state.quiz_attempts = 0
                st.rerun()
        
        # Feedback section
        if not st.session_state.feedback_submitted:
            with st.expander("📢 Provide Feedback"):
                st.write("Help us improve your learning experience!")
                feedback_text = st.text_area("What could we do better?")
                rating = st.slider("Rate your overall experience", 1, 5, 3)
                if st.button("Submit Feedback"):
                    feedback_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rating": rating,
                        "feedback": feedback_text
                    }
                    save_feedback(feedback_data)
                    st.session_state.feedback_submitted = True
                    st.success("Thank you for your feedback! 🙏")
        
        # Display feedback statistics (admin view)
        if st.checkbox("Show Feedback Statistics (Admin)"):
            feedback_data = load_feedback()
            if feedback_data:
                st.subheader("📊 Feedback Statistics")
                avg_rating = sum(f["rating"] for f in feedback_data) / len(feedback_data)
                st.write(f"Average Rating: {avg_rating:.2f} ⭐")
                
                st.write("Recent Feedback:")
                for feedback in feedback_data[-5:]:
                    st.write(f"**{feedback['timestamp']}** - Rating: {feedback['rating']} ⭐")
                    st.write(feedback['feedback'])
                    st.write("---")
            else:
                st.info("No feedback data available yet.")

if __name__ == "__main__":
    main()