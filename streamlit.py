import streamlit as st
import requests
st.title("Welcome to the Flask API! Ask your questions")

question=st.text_input("Enter your question:")

if st.button("ASK"):
    if question:
        url="http://127.0.0.1:5000/ask"
        response=requests.post(url,json={"question":question})
        answer=response.json().get("answer","No answer found")
        st.write(f"Answer: {answer}")
    else:
        st.write("Please enter a question")
