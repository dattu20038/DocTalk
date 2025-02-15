import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.7,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.15
        }
    )
    return llm

def main():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .footer-container {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: black;
            z-index: 9999;
            pointer-events: auto;
        }
        
        .custom-footer {
            color: white;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            margin-bottom: 0;
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
        
        .main-container {
            margin-bottom: 60px;
        }
        
        .stChatFloatingInputContainer {
            bottom: 60px !important;
        }
        
        .block-container {
            padding-bottom: 80px;
            margin-bottom: 60px;
        }
        
        .element-container {
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    with st.container():
        st.title("DocTalk – Your Digital Doctor, Always On Call!")
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        
        if prompt := st.chat_input("Pass your prompt here"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            CUSTOM_PROMPT_TEMPLATE = """
            Use the following pieces of context to answer the user's question about medical topics.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Keep your answers focused on the medical information provided in the context.
            
            Context: {context}
            Question: {question}
            
            Answer the question directly and professionally, like a medical professional would.
            Include relevant medical details from the context but explain them in clear terms.
            If the question is outside your medical knowledge base, state that clearly.
            """
            
            HUGGINGFACE_REPO_ID = "deepseek-ai/deepseek-7b-chat"
            HF_TOKEN = "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"
            
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
                
                with st.chat_message("assistant"):
                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]
                    source_documents = response["source_documents"]
                    
                    sources_text = "\n\n**Source Documents:**\n"
                    for i, doc in enumerate(source_documents, 1):
                        sources_text += f"{i}. {str(doc)}\n"
                    
                    result_to_show = f"{result}\n{sources_text}"
                    st.markdown(result_to_show)
                
                st.session_state.messages.append({"role": "assistant", "content": result_to_show})
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error("Please make sure your HF_TOKEN is correctly set in Streamlit secrets.")

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
