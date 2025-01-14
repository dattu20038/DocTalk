# DocTalk - Your Digital Doctor, Always On Call! 🏥

DocTalk is an intelligent medical chatbot powered by the Mistral-7B-Instruct model and LangChain. It provides accurate medical information by retrieving relevant context from uploaded medical documents using FAISS vector storage and semantic search.

## Features 

- Interactive chat interface built with Streamlit
- Document processing and semantic search using FAISS vector store
- Powered by Mistral-7B-Instruct large language model
- Contextual responses based on uploaded medical documents
- Source citation for transparency
- Easy-to-use interface for medical document uploads

## Technical Stack 

- **Language Model**: Mistral-7B-Instruct v0.3
- **Framework**: LangChain
- **Embeddings**: HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **Frontend**: Streamlit
- **Document Processing**: LangChain Document Loaders

## Installation 

1. Clone the repository:
```bash
git clone https://github.com/yourusername/doctalk.git
cd doctalk
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your HuggingFace API token:
   - Create a `.env` file in the project root
   - Add your HuggingFace API token:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```

## Usage 

1. Prepare your medical documents:
   - Place your PDF documents in the `data/` directory

2. Process the documents and create embeddings:
```bash
python process_documents.py
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

4. Access the application in your web browser at `http://localhost:8501`

## Project Structure 

```
doctalk/
├── app.py              # Main Streamlit application
├── process_documents.py # Document processing script
├── query_data.py       # Query handling script
├── data/              # Directory for PDF documents
├── vectorstore/       # FAISS vector store

```

## Configuration 

The following parameters can be adjusted in the code:

- `chunk_size`: Size of text chunks (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `temperature`: Model temperature (default: 0.5)
- `max_length`: Maximum response length (default: 512)
- `k`: Number of relevant chunks to retrieve (default: 3)

