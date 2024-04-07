import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import faiss
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, logo

def get_pdf_text(pdf_docs):
    """
    Read all the documents and concatinate the strings    
    """
    text = ""
    if len(pdf_docs) !=0:
        for pdf in pdf_docs:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text()
    else:
        st.write("No files Selected")
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="/n",
        chunk_size = 1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Docu Assistant", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Document Assistant for PDFs :books:")
    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace("{{MSG}}","Hello User"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello Bot"), unsafe_allow_html=True)
    

    with st.sidebar:
        st.write(logo, unsafe_allow_html=True)
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your pdf's here and Click on Process", accept_multiple_files=True)
        

        if st.button("Train"):
            with st.spinner("Processing"):
                # get the pdf files from upload button
                raw_text = get_pdf_text(pdf_docs)
                # create chunks
                text_chunks = get_text_chunks(raw_text)

                # Create Vector Store, you can use GPT api's or use one from hugging face.
                vectorstore = get_vectorstore(text_chunks)
                # Create conversational chain 
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                

if __name__ == "__main__":
    main()