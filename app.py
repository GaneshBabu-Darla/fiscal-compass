"""
Simple RAG App for Union Budget 2026-27
Ask questions about the budget document
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')

# Import libraries
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("🚀 Starting BudgetBuddy...")

# ========== STEP 1: LOAD PDF ==========
print("\n📄 Loading PDF...")
pdf_path = "data/raw/Union_Budget_Analysis-2026-27.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# ========== STEP 2: CHUNK TEXT ==========
print("\n✂️ Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# ========== STEP 3: CREATE EMBEDDINGS ==========
print("\n🔢 Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ========== STEP 4: CREATE VECTOR DATABASE ==========
print("\n💾 Creating vector database...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./data/chroma_db"
)
print("✅ Vector database ready!")

# ========== STEP 5: SETUP LLM ==========
print("\n🤖 Setting up Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1000
)

# ========== STEP 6: CREATE PROMPT ==========
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant answering questions about the Indian Union Budget 2026-27.
    Answer based ONLY on the provided context. If the answer isn't in the context, say "I don't have that information in the budget document."""),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

# ========== STEP 7: CREATE RAG CHAIN ==========
def get_answer(question):
    """Main function to answer questions"""
    
    # Search for relevant chunks
    results = vectordb.similarity_search(question, k=4)
    
    # Combine chunks into context
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Generate answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return answer, results

# ========== STEP 8: INTERACTIVE CHAT ==========
print("\n" + "="*50)
print("💬 BudgetBuddy is ready! Ask me anything about Union Budget 2026-27")
print("Type 'quit' to exit")
print("="*50 + "\n")

while True:
    question = input("\n❓ Your question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break
    
    if not question:
        continue
    
    print("\n🤔 Thinking...")
    
    try:
        answer, sources = get_answer(question)
        print(f"\n💡 Answer: {answer}")
        
        # Show sources (optional)
        show_sources = input("\n🔍 Show sources? (y/n): ").strip().lower()
        if show_sources == 'y':
            print("\n📚 Sources:")
            for i, doc in enumerate(sources[:2], 1):
                preview = doc.page_content[:200].replace('\n', ' ')
                print(f"{i}. ...{preview}...\n")
                
    except Exception as e:
        print(f"❌ Error: {e}")
