import os
import streamlit as st
from displayer import bot_template, user_template
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from ML.Ml import CityWeatherData

def init():
    load_dotenv()

    # Loading OpenAI API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set yet")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

def main():
    init()

    # Initialize LangChain Chat
    chat = ChatOpenAI(temperature=0)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
    st.header("Green Optimizer 🤖")

    with st.sidebar:
        # CHATTING CODE
        user_input = st.text_input("Enter your message:")
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking.."):
                if any(city in user_input.lower() for city in ["jeddah", "skaka", "riyadh", "dammam"]):
                    user_city = None
                    for city in ["jeddah", "skaka", "riyadh", "dammam"]:
                        if city in user_input.lower():
                            user_city = city
                            break

                    # If city name is found, proceed with solar energy analysis
                    if user_city:
                        # Display the form for user input
                        st.write(f"Welcome! Please enter the following information for {user_city.capitalize()}:")
                        with st.form(key='weather_form'):
                            ALLSKY = st.number_input("Enter ALLSKY:")
                            CLRSKY = st.number_input("Enter CLRSKY:")
                            pressure = st.number_input("Enter pressure:")
                            temperature = st.number_input("Enter temperature:")
                            moisture = st.number_input("Enter moisture:")

                            submit_button = st.form_submit_button(label='Submit')

                        # If the form is submitted, calculate and display the result
                        if submit_button:
                            # Create CityWeatherData instance
                            weather_data = CityWeatherData(city=user_city)

                            # Analyze weather for the selected city
                            solar_energy = weather_data.analyze_weather(
                                ALLSKY=ALLSKY, CLRSKY=CLRSKY, temperature=temperature, pressure=pressure, moisture=moisture
                            )
                            # Append the user and AI messages to the conversation
                            st.session_state.messages.append(AIMessage(content=f"The Solar Energy in {user_city.capitalize()} is: {solar_energy}"))

                    else:
                        # If no valid city name is found, proceed with regular chat
                        response = chat(st.session_state.messages)
                        st.session_state.messages.append(AIMessage(content=response.content))
                else:
                    # If no city name is mentioned, proceed with regular chat
                    response = chat(st.session_state.messages)
                    st.session_state.messages.append(AIMessage(content=response.content))

    # Displaying all the messages the user had by fetching them
    messages = st.session_state.get("messages", [])

    # Looping through all the messages, if 1 user(odd number) display it from the human position.
    # If two users(even number) display it from the bot position.
    for i, msgs in enumerate(messages[1:]):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()




'''import os#
import streamlit as st
from displayer import bot_template, user_template
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from ML.Ml import CityWeatherData

def init():
    load_dotenv()

    # Loading OpenAI API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set yet")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

def main():
    init()

    # Initialize LangChain Chat
    chat = ChatOpenAI(temperature=0)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Green Optimizer 🤖")

    with st.sidebar:
        # CHATTING CODE
        user_input = st.text_input("Enter your message:")
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking.."):
                if any(city in user_input.lower() for city in ["jeddah", "skaka", "riyadh", "dammam"]):
                    user_city = None
                    for city in ["jeddah", "skaka", "riyadh", "dammam"]:
                        if city in user_input.lower():
                            user_city = city
                            break

                    # If city name is found, proceed with solar energy analysis
                    if user_city:
                        # Prompt the user for additional information
                        ALLSKY = st.number_input("Enter ALLSKY:")
                        CLRSKY = st.number_input("Enter CLRSKY:")
                        pressure = st.number_input("Enter pressure:")
                        temperature = st.number_input("Enter temperature:")
                        moisture = st.number_input("Enter moisture:")

                        # Create CityWeatherData instance
                        weather_data = CityWeatherData(city=user_city)

                        # Analyze weather for the selected city
                        solar_energy = weather_data.analyze_weather(
                            ALLSKY=ALLSKY, CLRSKY=CLRSKY, temperature=temperature, pressure=pressure, moisture=moisture
                        )

                        # Display the solar energy result in a separate space
                        result_container = st.empty()
                        result_container.write(f"The Solar Energy in {user_city.capitalize()} is: {solar_energy}")

                        # Append the solar energy result to the conversation
                        st.session_state.messages.append(AIMessage(content=f"The Solar Energy in {user_city.capitalize()} is: {solar_energy}"))
                    else:
                        # If no valid city name is found, proceed with regular chat
                        response = chat(st.session_state.messages)
                        st.session_state.messages.append(AIMessage(content=response.content))
                else:
                    # If no city name is mentioned, proceed with regular chat
                    response = chat(st.session_state.messages)
                    st.session_state.messages.append(AIMessage(content=response.content))

    # Displaying all the messages the user had by fetching them
    messages = st.session_state.get("messages", [])

    # Looping through all the messages, if 1 user(odd number) display it from the human position.
    # If two users(even number) display it from the bot position.
    for i, msgs in enumerate(messages[1:]):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)

if __name__ == '__main__':
    main()

'''

























'''
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
'''