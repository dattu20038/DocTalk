import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Path to the FAISS vector store
DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache the vector store to avoid reloading it on every interaction
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Define a custom prompt template
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load the language model from Hugging Face
def load_llm(huggingface_repo_id, HF_TOKEN=None):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text2text-generation",  # Explicitly specify the task
        temperature=0.5,
        model_kwargs={
            "max_length": 512,
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 50,
        }
    )
    return llm

# Main function
def main():
    # Add custom CSS for the footer only
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Footer styles */
        .footer-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: black;
            z-index: 999;
        }
        
        .custom-footer {
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
        }
        
        .custom-footer a {
            color: white;
            text-decoration: none;
            transition: all 0.3s ease;
            padding: 5px 10px;
        }
        
        .custom-footer a:hover {
            color: #1E90FF;
            text-decoration: underline;
        }
        
        /* Add padding to main content to prevent footer overlap */
        .block-container {
            padding-bottom: 60px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("DocTalk – Your Digital Doctor, Always On Call!")
    
    # Initialize messages in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Pass your prompt here"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Don't provide anything out of the given context.
        Context: {context}
        Question: {question}
        Start the answer directly. No small talk please.
        """
        
        # Updated Model Configuration - Using T5 small for faster inference
        HUGGINGFACE_REPO_ID = "google/t5-small"
        HF_TOKEN = None  # Not required for this model
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )
            
            # Display assistant response
            with st.chat_message("assistant"):
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]
                result_to_show = result + "\nSource Docs:\n" + str(source_documents)
                st.markdown(result_to_show)
            
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Footer
    st.markdown(
        """
        <div class="footer-container">
            <div class="custom-footer">
                Designed by <a href="https://github.com/dattu20038" target="_blank"><b>Datta Srivathsava Gollapinni</b></a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
