import os
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

app = FastAPI()

# Global state for simplicity
vectorstore = None
llm = ChatOllama(model="llama3.2:3b", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class AnalyzeRequest(BaseModel):
    repo_path: str

class AskRequest(BaseModel):
    query: str

@app.post("/api/analyze")
def analyze_repo(req: AnalyzeRequest):
    global vectorstore
    repo_path = req.repo_path
    
    if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
        raise HTTPException(status_code=400, detail="Invalid directory path.")
        
    ALLOWED_EXTENSIONS = {'.py', '.js', '.jsx', '.md', '.json', '.txt'}
    IGNORE_DIRS = {'.git', 'venv', 'node_modules', '__pycache__', 'chroma_db', 'build', 'dist', '.next'}
    
    documents = []
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ALLOWED_EXTENSIONS or file in {'requirements.txt', 'package.json'}:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    rel_path = os.path.relpath(file_path, repo_path)
                    documents.append(Document(page_content=content, metadata={"source": rel_path}))
                except:
                    pass
                    
    if not documents:
        raise HTTPException(status_code=400, detail="No relevant files found in the repository.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    db_path = "./chroma_repo_db"
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection_name = "repo_guide"
    
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
        
    chroma_client.create_collection(name=collection_name)
    
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=embeddings
    )
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        ids = [f"chunk_{i+j}" for j in range(len(batch))]
        vectorstore.add_documents(documents=batch, ids=ids)
        
    return {"message": "Repository analyzed successfully.", "chunks": len(chunks)}


@app.post("/api/ask")
def ask_question(req: AskRequest):
    global vectorstore
    if not vectorstore:
        raise HTTPException(status_code=400, detail="Repository not analyzed yet. Please analyze a repository first.")
        
    query = req.query
    docs = vectorstore.similarity_search(query, k=5)
    
    if not docs:
        return {"answer": "No relevant information found in the repository.", "sources": []}
        
    sources = set()
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        sources.add(source)
        context_parts.append(f"--- File: {source} ---\n{doc.page_content}")
        
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are RepoGuide AI, an intelligent assistant for a code repository.
Answer the user's question ONLY using the supplied REPOSITORY CONTEXT.
If you don't know the answer based on the context, say so. Do not invent information.

CRITICAL INSTRUCTIONS:
1. Provide repository-specific answers based strictly on the retrieved source code.
2. When explaining a class, function, library, or component, explain its specific role and usage in THIS repository rather than providing a generic definition.
3. Keep your answer concise and direct.
4. Always mention the relevant source file names in your explanation.

REPOSITORY CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = llm.invoke(prompt)
    return {"answer": response.content, "sources": list(sources)}

# Mount static files and serve index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    print("\nStarting RepoGuide AI web server...")
    print("Open http://127.0.0.1:8000 in your browser to view the app.")
    uvicorn.run(app, host="127.0.0.1", port=8000)