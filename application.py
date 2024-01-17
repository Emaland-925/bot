import os
import streamlit as st
from displayer import bot_template, user_template
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)


def init():
    load_dotenv()

    #Loading openai api key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set yet")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")



def main():

    init()
    
    #Telling open ai the level of randomness the bot can reply with
    chat = ChatOpenAI(temperature = 0)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("ZIKO 🤖")
    
    with st.sidebar:
        #CHATTING CODE
        user = st.text_input("Enter your message to chat: ")
        if user:
            st.session_state.messages.append(HumanMessage(content=user))
            with st.spinner("Thinking.."):
                response = chat(st.session_state.messages)
                st.session_state.messages.append(AIMessage(content = response.content))
    
                
    #Displaying all of the messages the user had by fetching them
    #We used get to set a default value in case messages don't exist
    messages = st.session_state.get("messages", [])

    #Looping throw all of the messages if 1user(odd number) display it from the human position.
    #If two users(even number) display it from the bot position.
    for i, msgs in enumerate(messages[1:]):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()
