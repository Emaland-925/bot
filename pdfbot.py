import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from displayer import bot_template, user_template


from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS 
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI



def init():
    load_dotenv()

    #Loading openai api key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set yet")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

#A function to get the text from pdf files
def getting_pdf_text(pdfdocs):
    text = "" #User's raw text of the pdf files
    #Looping through files to read them then concatenate them to text
    for pdf in pdfdocs:
        pdf_reader = PdfReader(pdf)
        #Looping through pages to read them and get them in form of text
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#A function to split the text into chunks
def getting_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000, #size of the chunk
        chunk_overlap = 200, #To get full sentences
        length_function = len 
    )
    chunks = text_splitter.split_text(text)
    return chunks

#A function for embeddings
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

#A function to save the conversation history
def get_convo_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return convo_chain

#A function for getting the user questions
def user_input(userq):
    with st.spinner("Thinking.."):
        responses = st.session_state.convo({'question': userq})
        st.session_state.chat_history = responses['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():

    init()
    
    if "convo" not in st.session_state:
        st.session_state.convo = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ZIKO PDF 🤖")
 
    #PDF Bot actions
    userq = st.text_input("Ask your question about your documents here: ") 
    if userq:
        user_input(userq)

    with st.sidebar:
        
        st.subheader("Your documents")
        pdfdocs = st.file_uploader("Upload here you PDF then click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #Get the pdf text
                rawtext = getting_pdf_text(pdfdocs)
                #Getting the text chunks
                text_chunks = getting_chunks(rawtext)
                #Creating vector store using the chunks
                vectorstore = get_vectorstore(text_chunks)
                #Convo chain
                st.session_state.convo = get_convo_chain(vectorstore)

if __name__ == '__main__':
    main()