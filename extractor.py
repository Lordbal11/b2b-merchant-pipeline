import os
import warnings

# 1. Silence the warnings BEFORE importing anything else
os.environ["USER_AGENT"] = "langchain-scraper"
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load the Google API key from your .env file
load_dotenv()

def extract_lead_data(url):
    print(f"Scraping data from {url}...")
    
    # Download the webpage text
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_content = docs[0].page_content

    print("Extracting lead information using Gemini...")
    
    # Connect to Gemini's fast reasoning model
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Give Gemini strict instructions on what to extract
    prompt = PromptTemplate.from_template("""
    You are an expert sales intelligence assistant. Extract the following information from the website text below.
    If a piece of information is not found, write "Not found".

    Website Text:
    {text}

    Please format your response exactly like this:
    --- Extracted Lead Data ---
    Company: [Company Name]
    Emails: [All email addresses found]
    Phones: [All phone numbers found]
    What they do: [A 1-2 sentence summary of their value proposition]
    """)

    # Run the AI chain
    chain = prompt | llm
    response = chain.invoke({"text": text_content})
    
    return response.content

if __name__ == "__main__":
    # CHANGE THIS URL to a normal business website (like a local clinic or agency)
    # Example: target_url = "https://www.your-local-university.edu/contact"
    target_url =  "https://www.dramaranupam.com/contact-us" 
    
    result = extract_lead_data(target_url)
    print("\n" + result)