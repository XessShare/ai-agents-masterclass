from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
import os

load_dotenv()

model = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-405B-Instruct')
rag_directory = os.getenv('DIRECTORY', 'meeting_notes')

@st.cache_resource
def get_local_model():
    return HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=1024,
        do_sample=False
    )

    # If you want to run the model absolutely locally - VERY resource intense!
    # return HuggingFacePipeline.from_model_id(
    #     model_id=model,
    #     task="text-generation",
    #     pipeline_kwargs={
    #         "max_new_tokens": 1024,
    #         "top_k": 50,
    #         "temperature": 0.4
    #     },
    # )

llm = get_local_model()

def load_documents(directory):
    # Load the PDF or txt documents from the directory with file type filtering
    try:
        loader = DirectoryLoader(
            directory, 
            glob="**/*.txt",  # Only load text files for better performance
            show_progress=True
        )
        documents = loader.load()
        
        if not documents:
            print(f"Warning: No documents found in {directory}")
            return []

        # Split the documents into chunks with overlap for better context preservation
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,  # Add overlap to preserve context at boundaries
            length_function=len,
            separator="\n\n"  # Split on paragraph boundaries when possible
        )
        docs = text_splitter.split_documents(documents)
        
        print(f"Loaded {len(documents)} documents and created {len(docs)} chunks")
        return docs
        
    except Exception as e:
        print(f"Error loading documents from {directory}: {e}")
        return []

@st.cache_resource
def get_chroma_instance():
    # Get the documents split into chunks
    docs = load_documents(rag_directory)
    
    if not docs:
        print("No documents to process. Creating empty Chroma instance.")
        # Create an empty Chroma instance with the same embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(embedding_function=embedding_function)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    return Chroma.from_documents(docs, embedding_function)

db = get_chroma_instance()  

def query_documents(question):
    """
    Uses RAG to query documents for information to answer a question

    Example call:

    query_documents("What are the action items from the meeting on the 20th?")
    Args:
        question (str): The question the user asked that might be answerable from the searchable documents
    Returns:
        str: The list of texts (and their sources) that matched with the question the closest using RAG
    """
    try:
        # Check if database has any documents
        collection_count = db._collection.count()
        if collection_count == 0:
            return ["No documents available in the knowledge base."]
            
        similar_docs = db.similarity_search(question, k=5)
        
        if not similar_docs:
            return ["No relevant documents found for your question."]
            
        docs_formatted = list(map(
            lambda doc: f"Source: {doc.metadata.get('source', 'NA')}\nContent: {doc.page_content}", 
            similar_docs
        ))

        return docs_formatted
        
    except Exception as e:
        print(f"Error querying documents: {e}")
        return [f"Error occurred while searching documents: {str(e)}"]   

def prompt_ai(messages):
    # Fetch the relevant documents for the query
    user_prompt = messages[-1].content
    retrieved_context = query_documents(user_prompt)
    formatted_prompt = f"Context for answering the question:\n{retrieved_context}\nQuestion/user input:\n{user_prompt}"    

    # Prompt the AI with the latest user message
    doc_chatbot = ChatHuggingFace(llm=llm)
    ai_response = doc_chatbot.invoke(messages[:-1] + [HumanMessage(content=formatted_prompt)])

    return ai_response

def main():
    st.title("Chat with Local Documents")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"You are a personal assistant who answers questions based on the context provided if the provided context can answer the question. You only provide the answer to the question/user input and nothing else. The current date is: {datetime.now().date()}")
        ]    

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])        

    # React to user input
    # Example question: What's included in the wellness program Emily proposed?
    # Example question 2: What were the results of the team survey?
    # Example question 3: What was discussed in the meeting on the 22nd?
    if prompt := st.chat_input("What questions do you have?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            ai_response = prompt_ai(st.session_state.messages)
            st.markdown(ai_response.content)
        
        st.session_state.messages.append(ai_response)

if __name__ == "__main__":
    main()