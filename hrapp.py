import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS


# Function to read PDF content
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Mapping of PDFs
pdf_mapping = {
    'Handbook': 'Employee Handbook 2024.pdf',
    # Add more mappings as needed
}


# Load environment variables
load_dotenv()


# Main Streamlit app
def main():
    st.title("Z1 HR Team")
    with st.sidebar:
        st.title('🤗💬 HR Team')
        st.markdown('''
        ## About
        Choose the desired PDF, then perform a query.
        ''')


    custom_names = list(pdf_mapping.keys())

    selected_custom_name = st.sidebar.selectbox('Choose your PDF', ['', *custom_names])

    selected_actual_name = pdf_mapping.get(selected_custom_name)

    if selected_actual_name:
        pdf_folder = "pdfs"
        file_path = os.path.join(pdf_folder, selected_actual_name)

        try:
            text = read_pdf(file_path)
            st.info("Hello. I am Shivali. How can I help you?")
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return
        except Exception as e:
            st.error(f"Error occurred while reading the PDF: {e}")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        # Process the PDF text and create the documents list
        documents = text_splitter.split_text(text=text)

        # Vectorize the documents and create vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(documents, embedding=embeddings)

        st.session_state.processed_data = {
            "document_chunks": documents,
            "vectorstore": vectorstore,
        }

        # Load the Langchain chatbot
        llm = ChatOpenAI(temperature=0.5, max_tokens=1000, model_name="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Pre-define bot identity and purpose
        bot_identity = "Shivali, Chief of Staff at Z1 Tech"

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []


        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your questions from PDF "f'{selected_custom_name}'" using {bot_identity}:?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
            print(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            print(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()