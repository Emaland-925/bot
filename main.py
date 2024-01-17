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

# Importing functions for weather prediction and energy calculation from the file
from ML.Dammam import predict_weather_dammam
from ML.Riyadh import predict_weather_riyadh
from ML.SKAKA import predict_weather_skaka


def init():
    load_dotenv()

    # Loading the OpenAI API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set yet")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")


def extract_city_from_message(message):
    # Extracting the city name from the message (in this example, assuming it's the second word in the message)
    words = message.split()
    if len(words) > 1:
        return words[1]
    else:
        return None


def calculate_energy_production(weather_data):
    """
    Calculate energy production using weather data.

    Inputs:
    - weather_data: Values of weather data, may include information like temperature and wind speed.

    Output:
    - energy_production: Amount of energy produced based on weather data.
    """
    # Define the energy calculation algorithm here
    # This could be based on multiple factors like temperature and wind speed.

    # For this example, we'll calculate energy using temperature only as a simple factor.
    temperature = weather_data.get("temperature", 0)

    # Assume a simple linear relationship between temperature and energy production
    energy_production = temperature * 10  # Improve this relationship based on actual needs

    return energy_production


def main():
    init()

    # Set the level of randomness the bot can reply with
    chat = ChatOpenAI(temperature=0)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("ZIKO 🤖")

    with st.sidebar:
        # Chatting code
        user = st.text_input("Enter your message to chat: ")
        if user:
            st.session_state.messages.append(HumanMessage(content=user))
            with st.spinner("Thinking.."):
                # Use the city extraction algorithm
                user_city = extract_city_from_message(user)
                if user_city:
                    # Predict weather for the user-specified city
                    weather_prediction = predict_weather(user_city)
                    # Calculate energy production based on weather data
                    energy_production = calculate_energy_production(weather_prediction)
                    # Add weather predictions and energy production to the response with OpenAI
                    response = chat(st.session_state.messages, weather_prediction, energy_production)
                    st.session_state.messages.append(AIMessage(content=response.content))

    # Display all messages made by the user
    messages = st.session_state.get("messages", [])

    # Display based on order (individual messages shown from the human position, and pairs from the bot position)
    for i, msgs in enumerate(messages[1:]):
        if i % 2 == 0:
            st.markdown(user_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)
        else:
            st.markdown(bot_template.replace("{{MSG}}", msgs.content), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
