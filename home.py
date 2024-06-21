import streamlit as st

def show_home():
    st.title("Welcome to the Interview App")
    option = st.selectbox("Choose interview type:", ["Resume-Based", "Role-Based"])

    if st.button("Proceed"):
        if option == "Resume-Based":
            st.sidebar.page_link("pages/resume_based_interview.py")
        elif option == "Role-Based":
           st.sidebar.page_link("pages/role_based_interview.py")
        st.rerun()
