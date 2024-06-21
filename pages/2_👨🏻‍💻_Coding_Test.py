import streamlit as st
from streamlit_ace import st_ace
import io
import contextlib
import subprocess
from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
# load_dotenv()

# Initialize Hugging Face endpoint
# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

hf = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.5,
    huggingfacehub_api_token=HF_TOKEN,
    timeout=400
)

# Function to execute user code
def execute_code(code, language):
    result = io.StringIO()
    try:
        if language == 'python':
            with contextlib.redirect_stdout(result):
                exec(code)
        elif language == 'java':
            exec_java(code, result)
        elif language == 'c++':
            exec_cpp(code, result)
        else:
            result.write(f"Language {language} is not supported.")
    except Exception as e:
        result.write(str(e))
    return result.getvalue()

# Java code execution
def exec_java(code, output):
    try:
        with open("Solution.java", "w") as f:
            f.write(code)
        compile_result = subprocess.run(["javac", "Solution.java"], capture_output=True, text=True)
        if compile_result.returncode != 0:
            output.write(compile_result.stderr)
        else:
            run_result = subprocess.run(["java", "Solution"], capture_output=True, text=True)
            output.write(run_result.stdout)
            output.write(run_result.stderr)
    except Exception as e:
        output.write(str(e))

# C++ code execution
def exec_cpp(code, output):
    try:
        with open("solution.cpp", "w") as f:
            f.write(code)
        compile_result = subprocess.run(["g++", "solution.cpp", "-o", "solution"], capture_output=True, text=True)
        if compile_result.returncode != 0:
            output.write(compile_result.stderr)
        else:
            run_result = subprocess.run(["./solution"], capture_output=True, text=True)
            output.write(run_result.stdout)
            output.write(run_result.stderr)
    except Exception as e:
        output.write(str(e))

# Streamlit app
st.set_page_config(
    page_title="AI Interviewer - Coding Test",
    page_icon="💻",
)
with st.chat_message("assistant"):
    st.write("""
🚀 **Welcome to AI Interviewer's Coding Test!**

Get ready to showcase your coding skills. You will be presented with two coding questions. 
Write your code in the editor provided for each question. After writing your code, apply and run to see the output.

To proceed to the next question, click on **Submit & Feedback** to receive detailed insights.

To start the interview Click on the Start Button ▶️.
""")

# Initialize interview questions
prompt_template_1 = PromptTemplate.from_template(
    template="""Generate only one hard level coding interview question.
    Questions can be from any of the following topics: Arrays, String, Graph, Tree, LinkedList, Stack, Queue.
    Do not include any approaches, brief, or introduction about the question.
    Provide the question with the constraints.
    
Question 1:
"""
)

prompt_template_2 = PromptTemplate.from_template(
    template="""Generate only one hard level coding interview question.
    Questions can be from any of the following topics: Arrays, String, Graph, Tree, LinkedList, Stack, Queue.
    Do not include any approaches, brief, or introduction about the question.
    Provide the question with the constraints.
    
Question 2:
"""
)

# Generate the first question from the model
prompt_str_1 = prompt_template_1.format()
response_1 = hf.predict(text=prompt_str_1)
question_1 = response_1.strip()

# Generate the second question from the model
prompt_str_2 = prompt_template_2.format()
response_2 = hf.predict(text=prompt_str_2)
question_2 = response_2.strip()

# Create the questions list
questions = [question_1, question_2]

# Feedback prompt template
prompt_template_feedback = """
Evaluate the candidate's response to the following interview question:
{question}

Candidate's Response:
{response}

Feedback: provide the feedback and also include sample answer
{feedback}
"""

# Initialize session state
if 'question_index' not in st.session_state:
    st.session_state.question_index = -1  # Start with -1 to indicate that the start button hasn't been clicked

if 'code_storage' not in st.session_state:
    st.session_state.code_storage = {'python': {}, 'java': {}, 'c++': {}}

# Ensure the structure of code_storage is maintained
if not isinstance(st.session_state.code_storage, dict):
    st.session_state.code_storage = {'python': {}, 'java': {}, 'c++': {}}

for lang in ['python', 'java', 'c++']:
    if lang not in st.session_state.code_storage or not isinstance(st.session_state.code_storage[lang], dict):
        st.session_state.code_storage[lang] = {}

# Supported languages
languages = ['python', 'java', 'c++']

# Display start button if no question has been started
if st.session_state.question_index == -1:
    if st.button("Start"):
        st.session_state.question_index = 0  # Move to the first question

# Display current question and editor
if st.session_state.question_index >= 0 and st.session_state.question_index < len(questions):
    current_question = questions[st.session_state.question_index]
    st.write(f"Question {st.session_state.question_index + 1}: {current_question}")

    # Dropdown to select language
    selected_language = st.selectbox("Select Language", languages)

    # Retrieve saved code for the current question and selected language
    saved_code = st.session_state.code_storage[selected_language].get(st.session_state.question_index, "")

    # ACE code editor for user to input their code
    user_code = st_ace(
        value=saved_code,
        placeholder="Enter your code here...",
        language=selected_language if selected_language != 'c++' else 'c_cpp',  # Use 'c_cpp' for C++ mode
        theme='monokai',
        key=f'ace_editor_{selected_language}_{st.session_state.question_index}'
    )

    # Button to submit code
    if st.button("Run"):
        if user_code.strip() != "":
            # Save the code to session state
            st.session_state.code_storage[selected_language][st.session_state.question_index] = user_code
            # Execute the code
            output = execute_code(user_code, selected_language)
            st.subheader("Output")
            st.code(output)
        else:
            st.warning("Please enter some code before submitting.")

    # Button to submit answer and receive feedback
    if st.button("Submit Answer for Feedback"):
        if user_code.strip() != "":
            # Save the code to session state
            st.session_state.code_storage[selected_language][st.session_state.question_index] = user_code
            # Execute the code
            output = execute_code(user_code, selected_language)
            st.subheader("Output")
            st.code(output)
            # Generate feedback
            feedback_prompt = PromptTemplate(template=prompt_template_feedback, input_variables=["question", "response", "feedback"])
            feedback_query = feedback_prompt.format(question=current_question, response=user_code, feedback="")
            feedback_response = hf.predict(text=feedback_query)
            st.subheader("Feedback")
            st.write(feedback_response)
            # Mark question as answered
            st.session_state[f'answered_{st.session_state.question_index}'] = True
        else:
            st.warning("Please enter some code before submitting.")

    # Button to move to the next question
    if st.session_state.question_index < len(questions) - 1:
        if st.session_state.get(f'answered_{st.session_state.question_index}', False):
            if st.button("Next Question"):
                st.session_state.question_index += 1
                st.session_state[f'answered_{st.session_state.question_index}'] = False  # Reset next question state
                st.rerun()
        else:
            st.warning("Please submit your answer and receive feedback before moving on to the next question.")

    # Button to reset the interview
    if st.button("Reset Interview"):
        st.session_state.question_index = -1  # Reset to initial state
        st.session_state.code_storage = {'python': {}, 'java': {}, 'c++': {}}
        st.rerun()

# Display success message only if all questions are completed
if st.session_state.question_index == len(questions):
    st.success("**You have completed all the questions!**")

