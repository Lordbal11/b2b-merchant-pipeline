import os
import time
import csv
from datetime import datetime

# 1. Set Environment Variables
# Prevents WebBaseLoader user-agent warnings and sets up your Gemini API key
os.environ["USER_AGENT"] = "MyBatchExtractor/1.0"
os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6KB_XHegTbU0qbjJsi3l6cOqkeDsDCh2916SKCQqEsAuw"

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Output CSV configuration
csv_filename = "B2B_Financial_Offerings_Extract.csv"

# Initialize CSV and write header row
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Extraction Date", "Company URL", "Core Financial Offerings (AI Summary)", "Status"])

print(f"🚀 Starting B2B data extraction pipeline...\nSaving cleanly formatted results to: {csv_filename}\n")

# Target list of 15 Delhi NCR financial companies & NBFCs
urls = [
    "https://www.lendingkart.com",
    "https://www.progcap.com",
    "https://www.oxyzo.in",
    "https://www.biz2credit.in",
    "https://www.stashfin.com",
    "https://www.ayefin.com",
    "https://www.faircent.in",
    "https://www.indifi.com",
    "https://axio.co.in",
    "https://ziploan.in",
    "https://kinaracapital.com",
    "https://www.herofincorp.com",
    "https://www.mswipe.com",
    "https://flexiloans.com",
    "https://www.incred.com"
]

# Initialize the Gemini model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
except Exception as e:
    print(f"Failed to initialize the Gemini model: {e}")
    exit(1)

for url in urls:
    print(f"🔄 Processing: {url}")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Load website content
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        if not docs:
            raise ValueError("No content was scraped from the website.")
            
        # Limit scraped text length to keep prompt within bounds
        page_content = docs[0].page_content[:5000] 
        prompt = f"Analyze the following website content and provide a brief, 2-sentence summary of the company's core financial offerings:\n\n{page_content}"
        
        # Invoke Gemini AI
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Extract text content safely (handles both string and list responses)
        raw_content = response.content
        if isinstance(raw_content, list):
            text_result = " ".join([str(block.get("text", "")) if isinstance(block, dict) else str(block) for block in raw_content])
        else:
            text_result = str(raw_content)
            
        # ------------------------------------------------------------------
        # CLEANING LOGIC: Strips line breaks, returns, and extra spaces
        # ------------------------------------------------------------------
        clean_result = text_result.replace("\n", " ").replace("\r", " ").strip()
        clean_result = " ".join(clean_result.split())
        # ------------------------------------------------------------------

        print(f"   ✅ Success: {clean_result[:80]}...")
        
        # Append clean data row to CSV
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, url, clean_result, "SUCCESS"])
            
    except Exception as e:
        error_msg = str(e).replace("\n", " ").replace("\r", " ").strip()
        error_msg = " ".join(error_msg.split())
        
        print(f"   ❌ Failed to extract data for {url}")
        print(f"      [SYSTEM ERROR]: {error_msg}\n")
        
        # Log failure to CSV without breaking file structure
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, url, error_msg, "FAILED"])
        
    # Rate limit spacing between requests
    try:
        time.sleep(15)
    except KeyboardInterrupt:
        print("\n🛑 Extraction manually interrupted by user. Saved all data processed so far.")
        break

print(f"\n🎉 Pipeline complete! Deliverable ready at: {csv_filename}")