import os
os.environ["USER_AGENT"] = "langchain-scraper"
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load API key (automatically finds GOOGLE_API_KEY)
load_dotenv()

def build_rag_chatbot(url: str):
    print(f"Scraping data from {url}...")
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("Building vector database with Google Embeddings...")
    # Using Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("Initializing Gemini LLM...")
    # Using the latest Gemini Flash model for speed and efficiency
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    target_url = "https://en.wikipedia.org/wiki/Wikipedia:About"
    chatbot = build_rag_chatbot(target_url)
    
    while True:
        user_input = input("\nAsk the chatbot a question (or type 'quit'): ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        response = chatbot.invoke({"input": user_input})
        print(f"\nAnswer: {response['answer']}")