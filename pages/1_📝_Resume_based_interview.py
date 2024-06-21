import streamlit as st
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# import os

# load_dotenv()

# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_TOKEN = st.secrets["HUGGINGFACE_ACCESS_TOKEN"]
st.title("Resume-Based Interview")
with st.chat_message("assistant"):
       st.write("""
        👋 **Hello, I’m your AI Interviewer!**

        Ready to embark on your interview preparation journey? Let's get started! 

        To begin, upload your resume in PDF FORMAT:
        """)
st.write('Please upload your resume in PDF format')

# Initialize session state attributes
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'technical_questions' not in st.session_state:
    st.session_state.technical_questions = []
if 'behavioural_questions' not in st.session_state:
    st.session_state.behavioural_questions = []
if 'coding_questions' not in st.session_state:
    st.session_state.coding_questions = []
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0
if 'current_questions' not in st.session_state:
    st.session_state.current_questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'question_displayed' not in st.session_state:
    st.session_state.question_displayed = False
if 'current_interview_type' not in st.session_state:
    st.session_state.current_interview_type = None
if 'in_interview' not in st.session_state:
    st.session_state.in_interview = False

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.success("Successfully uploaded the resume")
    save_folder = 'resume'
    save_path = Path(save_folder, uploaded_file.name)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    with st.chat_message("assistant"):
       st.write("""

       Great!!You successfully uploaded the resume so to get started with the interview,
        please select the type of interview you'd like to practice:
        """)


    # Define buttons and actions
    submit1 = st.button("Technical Interview")
    submit2 = st.button("Behavioural Interview")
    submit3 = st.button("Interview based on DSA Fundamentals")

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
        timeout=400
    )

    loader = PyPDFLoader(save_path)
    pdf_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(pdf_docs)

    vectorstore = FAISS.from_documents(final_documents[:1000], huggingface_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    prompt_template_technical = """
Based on the following resume content and the role, generate a detailed list of 7 potential technical interview questions that an interviewer might ask the candidate. Ensure the questions comprehensively cover various aspects such as the candidate's technical skills, projects, and experiences. Provide the questions in a numbered list format.

Resume Content:
{context}

Questions:
1.
2.
3.
4.
5.
6.
7.
"""

    prompt_template_behavioural = """
Based on the following resume content and the role, generate a detailed list of 7 potential behavioural interview questions that an interviewer might ask the candidate. Ensure the questions comprehensively cover various aspects such as the candidate's soft skills, teamwork, leadership, and problem-solving abilities. Provide the questions in a numbered list format.

Resume Content:
{context}

Questions:
1.
2.
3.
4.
5.
6.
7.
"""

    prompt_template_coding = """
Generate a detailed list of 5 unique and potential theory-based data structures and algorithms interview questions that an interviewer might ask the candidate. Ensure the questions comprehensively cover various aspects of the candidate's understanding and fundamentals of data structures and algorithms, including arrays, strings, linked lists, graphs, trees, stacks, queues, and time complexity. Make sure you generate different questions every time. The questions must be theory-based, discussing the approach rather than asking to code or implement. Provide the questions in a numbered list format.

Resume Content:
{context}

Questions:
1. 
2. 
3. 
4. 
5. 
"""


    prompt_template_feedback = """
Evaluate the candidate's response to the following interview question:
{question}

Candidate's Response:
{response}

Feedback:provide the feedback and also include sample answer
{feedback}
"""

    # Handle button clicks and reset session state for new interviews
    if submit1 or submit2 or submit3:
        if st.session_state.in_interview:
            st.warning("Please complete the current interview before starting a new one.")
        else:
            context = "\n".join([doc.page_content for doc in final_documents])

            if submit1:
                prompt_template = prompt_template_technical
                session_key = 'technical_questions'
                interview_type = 'technical'
            elif submit2:
                prompt_template = prompt_template_behavioural
                session_key = 'behavioural_questions'
                interview_type = 'behavioural'
            elif submit3:
                prompt_template = prompt_template_coding
                session_key = 'coding_questions'
                interview_type = 'coding'

            # Reset session state for new interview type
            if st.session_state.current_interview_type != interview_type:
                st.session_state.messages = []
                st.session_state.answers = []
                st.session_state.question_index = 0
                st.session_state.current_questions = []
                st.session_state.question_displayed = False
                st.session_state.current_interview_type = interview_type

            prompt = PromptTemplate(template=prompt_template, input_variables=["context"])

            retrievalQA = RetrievalQA.from_chain_type(
                llm=hf,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )

            query = prompt.format(context=context)
            result = retrievalQA({"query": query})
            interview_questions = result['result'].strip().split('\n')

            st.session_state[session_key] = interview_questions

            st.session_state['question_index'] = 0
            st.session_state['current_questions'] = interview_questions
            st.session_state['question_displayed'] = False
            st.session_state.in_interview = True

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input and display questions one by one
if st.session_state.get('current_questions'):
    question_index = st.session_state['question_index']
    if question_index < len(st.session_state['current_questions']):
        current_question = st.session_state['current_questions'][question_index].strip()
        if current_question and not st.session_state['question_displayed']:
            st.session_state.messages.append({"role": "assistant", "content": current_question})
            st.session_state.question_displayed = True

        with st.chat_message("assistant"):
            st.markdown(current_question)

        if user_input := st.chat_input("Your answer:"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state['answers'].append(user_input)
            st.session_state['question_index'] += 1
            st.session_state.question_displayed = False
            st.experimental_rerun()
    else:
        

        st.write("🌟 Congratulations! You have completed the interview.")

        st.write("""
            Thank you for sharing your insights with me! 🎤✨

            I will now provide detailed feedback for each question. Sit tight and let's review your performance together.
    """)

        # st.write("Answers:")
        # for i, ans in enumerate(st.session_state['answers']):
        #     st.write(f"Question {i + 1}:")
        #     st.write(ans)
        st.session_state.in_interview = False

        # Generate and display feedback for each question
        if st.session_state['current_interview_type'] == 'technical':
            session_key = 'technical_questions'
        elif st.session_state['current_interview_type'] == 'behavioural':
            session_key = 'behavioural_questions'
        elif st.session_state['current_interview_type'] == 'coding':
            session_key = 'coding_questions'

        interview_questions = st.session_state.get(session_key, [])
        candidate_responses = st.session_state['answers']

        if interview_questions and candidate_responses:
            st.write("**Feedback on Your Responses:**")

            for i, (question, response) in enumerate(zip(interview_questions, candidate_responses)):
                prompt_feedback = PromptTemplate(template=prompt_template_feedback, input_variables=["question", "response", "feedback"])
                feedback_query = prompt_feedback.format(question=question, response=response, feedback="")

                feedback_response = hf.predict(text=feedback_query)

                st.write(f"Question {i + 1}: {question}")
                st.write(f"Your Response: {response}")
                st.write("Feedback:")
                st.write(feedback_response)

# Display stored questions
# (your existing code to display stored questions)
