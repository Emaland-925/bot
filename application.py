import os
import streamlit as st
from displayer import bot_template, user_template
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from ML.Ml import CityWeatherData
from langchain import globals

# Set verbose
globals.set_verbose(True)

# Get verbose
verbose = globals.get_verbose()

# Set debug
globals.set_debug(True)

# Get debug
debug = globals.get_debug()

# Set llm_cache
globals.set_llm_cache(True)

# Get llm_cache
llm_cache = globals.get_llm_cache()



def init():
    load_dotenv()

    # Loading OpenAI API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set yet")
        st.error('ERROR: API KEY IS NOT SET')
    else:
        print("OPENAI_API_KEY is set")
    
    st.set_page_config(
        page_title="Green Optimizer",
        page_icon="❇️",
    )


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
        cities = "jeddah", "skaka", "riyadh", "dammam"

        if user_input:
            #st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking.."):
                if any(city in user_input.lower() for city in cities):
                    user_city = None
                    for city in cities:
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
                            moisture = st.number_input("Enter relative humidity:")

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
                            st.session_state.messages.append(HumanMessage(content=user_input))
                            st.session_state.messages.append(AIMessage(content=f"The Solar Energy in {user_city.capitalize()} is: {solar_energy}"))


                #If the city not from what we have
                else:
                    st.markdown(bot_template.replace("{{MSG}}", "We do not have enough information about this city yet."), unsafe_allow_html=True)
                    
     
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
