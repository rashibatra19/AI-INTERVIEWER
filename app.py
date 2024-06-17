from pathlib import Path
import streamlit as st
from langchain.chains import ConversationChain, RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import time
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from streamlit_chat import message
# import os 
# from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# load_dotenv()
# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# ANTHROPIC_KEY=os.getenv("ANTHROPIC_KEY")
HF_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]
st.write('# AI Interviewer')
st.write("This AI-powered application helps users prepare for job interviews by generating tailored interview questions based on their uploaded resumes in PDF format. Users can answer these questions one by one. The app evaluates their responses, providing constructive feedback on how to improve answers and assigns a score out of 100 to evaluate their overall interview performance.")
st.write('Please upload your resume in pdf format')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.success("Successfully uploaded the resume")
    save_folder = 'resume'
    save_path = Path(save_folder, uploaded_file.name)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())

    # Directly read from the PDF
    loader = PyPDFLoader(save_path)
    pdf_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(pdf_docs)

    # Embedding using HuggingFace
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    hf = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
    )

    # Vectorstore creation
    vectorstore = FAISS.from_documents(final_documents[:1000], huggingface_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template = """
Based on the following resume content, generate a detailed list of 10 potential interview questions that an interviewer might ask the candidate. Ensure the questions comprehensively cover various aspects such as the candidate's skills, projects, experience, and HR-related questions. Provide the questions in a numbered list format, including a mix of 5 medium questions and 5 advanced questions.

    Resume Content:
    {context}

    Here are some questions that can be asked in the interview based on your resume:
    
    Questions:
    
    1.
    2.
    3.
    4.
    5.
    6.
    7.
    8.
    9.
    10.
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    query = "Generate a detailed list of interview questions based on the resume provided. Ensure that the questions comprehensively cover various aspects such as the candidate's skills, projects, experience, and qualifications and also some HR QUESTIONS. Provide the questions in a numbered list format. Include a mix of 5 medium questions and 5 advanced questions, clearly categorized as indicated in the prompt template."

    result = retrievalQA.invoke({"query": query})
    interview_questions = result['result'].strip().split('\n')

    # st.write("Based on your resume, here are some potential interview questions that you might be asked:")
    # st.write(interview_questions)

    # Initialize conversation memory
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0

    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = []

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    # Store interview questions in memory
    st.session_state.buffer_memory.save_context(
        {"input": "Generated interview questions"},
        {"output": interview_questions}
    )

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    chat_prompt_template = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), human_msg_template])


    conversation_llm = HuggingFaceEndpoint(repo_id=repo_id, huggingfacehub_api_token=HF_TOKEN, max_length=128, temperature=0.5)
    # conversation_llm=ChatAnthropic(model='claude-3-opus-20240229',api_key=ANTHROPIC_KEY)
    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=chat_prompt_template, llm=conversation_llm, verbose=True)

    response_container = st.container()
    textcontainer = st.container()

    user_response = None
    if st.session_state.current_question_index < len(interview_questions):
        current_question = interview_questions[st.session_state.current_question_index]
        with textcontainer:
            st.write(f"Question {st.session_state.current_question_index + 1}: {current_question}")
            user_response = st.text_input("Your answer: ", key=f"input_{st.session_state.current_question_index}")
    else:
        current_question = None

    with response_container:
        if st.session_state['user_responses']:
            for i, (question, response) in enumerate(zip(interview_questions, st.session_state['user_responses'])):
                message(f"Q{i + 1}: {question}", key=f"q{i}")
                message(f"Your answer: {response}", key=f"a{i}", is_user=True)

    if user_response is not None and user_response != "":
        if st.session_state.get("last_response", "") != user_response:
            # Save the last response to avoid re-processing the same input
            st.session_state["last_response"] = user_response

            # Save the user's response
            st.session_state.user_responses.append(user_response)

            # Move to the next question
            st.session_state.current_question_index += 1

            st.rerun()

    if current_question is None and st.session_state.user_responses:
        st.write("You have answered all the questions. Here is the feedback on your responses:")

        combined_responses = "\n".join([f"Q{i + 1}: {q}\nYour answer: {a}" for i, (q, a) in enumerate(zip(interview_questions, st.session_state['user_responses']))])
    
    # Splitting the responses into two parts
        combined_responses_1 = combined_responses[:len(interview_questions) // 2]
        combined_responses_2 = combined_responses[len(interview_questions) // 2:]

        feedback_query_1 = f"Evaluate the following responses and provide brief feedback on questions that are not answered well, with a sample example and suggestions for improvement:\n\n{combined_responses_1}\n\nPlease evaluate the candidate's performance for these questions."
        feedback_response_1 = conversation.predict(input=feedback_query_1)
        # st.write(feedback_response_1)
        # time.sleep(2)
        feedback_query_2 = f"Evaluate the following responses and provide brief feedback on questions that are not answered well, with a sample example and suggestions for improvement:\n\n{combined_responses_2}\n\nPlease evaluate the candidate's performance for these questions."
        feedback_response_2 = conversation.predict(input=feedback_query_2)
        # st.write(feedback_response_2)
        
    # Combining the feedback responses
        combined_feedback = f"{feedback_response_1}\n\n{feedback_response_2}"
        # st.write(combined_feedback)

    # Generating a final summary and feedback report
        final_feedback_query = f"""Based on this feedback: {combined_feedback} , provide a final summary report. 
        Include a specific areas of improvements.Finally 
        assign a score out of 100 for the candidate's interview performance and not for single 
        question-answer"""
        final_feedback_response = conversation.predict(input=final_feedback_query)

        st.write("Final Feedback Report:")
        st.write(final_feedback_response)




        # with st.spinner("Generating feedback..."):
        #     feedback_response = conversation.predict(input=feedback_query)

        # st.write(feedback_response)

        # Reset the session state to allow re-answering
        if st.button("Restart"):
            st.session_state.current_question_index = 0
            st.session_state.user_responses = []
            st.session_state.last_response = ""
            st.rerun()
