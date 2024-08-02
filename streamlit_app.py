import streamlit as st
from chatbot_functions import preprocess, chatbot

# Load and preprocess text
with open('/content/oil_and_gas_kids.txt', 'r') as file:
    text = file.read()
sentences = preprocess(text)

# Streamlit app
st.title('Oil and Gas Chatbot')

user_query = st.text_input("Ask a question about oil and gas:")

if user_query:
    response = chatbot(user_query, sentences)
    st.write("Chatbot Response:")
    st.write(response)
