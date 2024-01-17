import streamlit as st
from application import main as application_main
from pdfbot import main as pdfbot_main

st.set_page_config(
    page_title="ZIKO",
    page_icon="🤖"
)

# Center-align the select box outside the sidebar
st.title("Choose your Bot")

st.markdown(
    """
    <style>
    .css-17zskas {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Creating a menu to choose the bot
selected_bot = st.selectbox(' ', ('ZIKO BOT', 'ZIKO PDF BOT'))

if selected_bot == 'ZIKO BOT':
    application_main()
    
elif selected_bot == 'ZIKO PDF BOT':
    pdfbot_main()
