import os
import streamlit as str_web
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Configure the Streamlit UI Page Layout
str_web.set_page_config(page_title="Corporate Brain AI", page_icon="🧠", layout="centered")
str_web.title("🧠 Enterprise Internal Brain AI")
str_web.subheader("Secure Knowledge Retrieval Infrastructure")

# Define a persistent directory to store our embedded data safely
DB_DIR = "corporate_chroma_db"

# Initialize Session State variables to track if the data engine is ready
if "engine_ready" not in str_web.session_state:
    str_web.session_state.engine_ready = False

# --- FRONTEND UI: FILE UPLOADER SIDEBAR ---
with str_web.sidebar:
    str_web.header("📁 Knowledge Ingestion")
    uploaded_file = str_web.file_uploader("Upload Company Manual (PDF Only)", type=["pdf"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily to disk so our loader can read it
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if str_web.button("Process & Train Brain"):
            with str_web.spinner("Engineering vector space... Please wait."):
                try:
                    # 1. Load the PDF data
                    loader = PyPDFLoader(temp_file_path)
                    docs = loader.load()
                    
                    # 2. Chunk text into optimal semantic windows
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
                    
                    # 3. Embed text into mathematical vector spaces using Gemini
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
                    
                    str_web.session_state.engine_ready = True
                    str_web.success("💥 Brain Trained Successfully! Engine Ready.")
                except Exception as e:
                    str_web.error(f"Ingestion failed: {e}")
                finally:
                    # Clean up temporary file from local storage
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

# --- MAIN UI: CONVERSATIONAL AGENT INTERFACE ---
if str_web.session_state.engine_ready:
    str_web.info("System Status: Online. Ask questions regarding ingested enterprise data below.")
    
    # Text input for user query
    user_query = str_web.text_input("Enter your query for the Corporate Brain:")
    
    if user_query:
        with str_web.spinner("Querying vector index..."):
            # Reconnect to the stored Vector database
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Setup enterprise-grade security prompt to completely eliminate hallucinations
            system_prompt = (
                "You are an expert enterprise corporate assistant. Answer the user's question using ONLY the provided context below.\n"
                "If you do not know the answer based on the context, state strictly: 'I cannot find the answer within company documentation.'\n"
                "Do not invent facts outside the text.\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            # Build and invoke the RAG architecture pipeline
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            response = rag_chain.invoke({"input": user_query})
            
            # Output the answer beautifully in a clean UI box
            str_web.markdown("### 💬 System Response:")
            str_web.write(response["answer"])
else:
    str_web.warning("👈 Please upload and process a company PDF manual in the sidebar to power on the engine.")