import os
import chromadb

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


# ==================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError(
        "TAVILY_API_KEY not found in .env file."
    )


# ==================================================
# 2. LOAD LOCAL KNOWLEDGE BASE
# ==================================================

loader = TextLoader("knowledge.txt")

documents = loader.load()

print("Local knowledge base loaded successfully.")


# ==================================================
# 3. SPLIT DOCUMENT INTO CHUNKS
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(
    documents
)

print(
    "Number of chunks:",
    len(chunks)
)


# ==================================================
# 4. CREATE EMBEDDINGS
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name=(
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )
)

print("Embedding model loaded.")


# ==================================================
# 5. CREATE PERSISTENT CHROMADB
# ==================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = (
    chroma_client.get_or_create_collection(
        name="research_knowledge"
    )
)


vectorstore = Chroma(
    client=chroma_client,
    collection_name="research_knowledge",
    embedding_function=embeddings
)


# Add documents only once
if collection.count() == 0:

    ids = [
        f"research_{i}"
        for i in range(len(chunks))
    ]

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    print(
        "Research knowledge added to ChromaDB."
    )

else:

    print(
        "Existing ChromaDB loaded.",
        "Documents:",
        collection.count()
    )


# ==================================================
# 6. CREATE LOCAL LLM
# ==================================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

print("Local LLM loaded.")


# ==================================================
# 7. CREATE TAVILY WEB SEARCH
# ==================================================

web_search_tool = TavilySearch(
    max_results=3,
    topic="general"
)

print("Tavily Web Search loaded.")


# ==================================================
# 8. LOCAL VECTOR SEARCH
# ==================================================

def vector_search(query):

    print(
        "\n[CHROMADB SEARCH CALLED]"
    )

    docs = vectorstore.similarity_search(
        query,
        k=3
    )

    if not docs:
        return ""

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context


# ==================================================
# 9. CHECK WHETHER LOCAL CONTEXT IS ENOUGH
# ==================================================

def check_context(
    query,
    context
):

    print(
        "\n[CONTEXT EVALUATOR CALLED]"
    )


    if not context:
        return "NO"


    prompt = f"""
You are a strict knowledge-base evaluator.

Your job is to determine whether the LOCAL CONTEXT
contains enough information to answer the user's
question.

RULES:

1. Answer YES only when the local context directly
   contains enough information.

2. Answer NO when the required information is missing.

3. Answer NO for current or real-time questions such as:
   - latest
   - today
   - current
   - recent
   - news
   - live
   - this week
   - this month
   - latest developments

4. Do NOT answer the user's question.

5. Do NOT use your own knowledge.

6. Return ONLY one word:

YES

or

NO


LOCAL CONTEXT:
{context}


USER QUESTION:
{query}


DECISION:
"""


    response = llm.invoke(
        prompt
    )


    decision = (
        response.content
        .strip()
        .upper()
    )


    if decision.startswith("YES"):
        return "YES"

    return "NO"


# ==================================================
# 10. ANSWER USING LOCAL KNOWLEDGE
# ==================================================

def answer_from_local_kb(
    query,
    context
):

    print(
        "\n[USING LOCAL KNOWLEDGE BASE]"
    )


    prompt = f"""
You are an AI Research Assistant.

Answer the user's question ONLY using
the supplied LOCAL KNOWLEDGE BASE.

Do not use outside knowledge.

Do not invent additional information.

If the answer cannot be found in the context,
say:

"I could not find enough information in
the local knowledge base."


LOCAL KNOWLEDGE BASE:
{context}


QUESTION:
{query}


ANSWER:
"""


    response = llm.invoke(
        prompt
    )


    return response.content


# ==================================================
# 11. TAVILY WEB SEARCH
# ==================================================

def web_search(query):

    print(
        "\n[TAVILY WEB SEARCH CALLED]"
    )


    try:

        result = web_search_tool.invoke({
            "query": query
        })

        return result


    except Exception as error:

        return {
            "error": str(error)
        }


# ==================================================
# 12. FORMAT WEB RESULTS
# ==================================================

def format_web_results(result):

    if not result:

        return (
            "No web search results were available."
        )


    # ----------------------------------------------
    # Error case
    # ----------------------------------------------

    if (
        isinstance(result, dict)
        and "error" in result
    ):

        return (
            "Web search error: "
            + result["error"]
        )


    # ----------------------------------------------
    # Tavily dictionary response
    # ----------------------------------------------

    if isinstance(result, dict):

        results = result.get(
            "results",
            []
        )


        if not results:

            return str(result)


        formatted_results = []


        for item in results:

            title = item.get(
                "title",
                "Untitled Source"
            )

            url = item.get(
                "url",
                ""
            )

            content = item.get(
                "content",
                ""
            )


            formatted_results.append(
                f"""
SOURCE TITLE:
{title}

SOURCE URL:
{url}

CONTENT:
{content}
""".strip()
            )


        return "\n\n".join(
            formatted_results
        )


    # ----------------------------------------------
    # Other return types
    # ----------------------------------------------

    return str(result)


# ==================================================
# 13. GENERATE ANSWER USING WEB RESULTS
# ==================================================

def answer_from_web(
    query,
    web_context
):

    print(
        "\n[GENERATING WEB-BASED ANSWER]"
    )


    prompt = f"""
You are an AI Research Assistant.

Answer the user's question ONLY using
the supplied WEB SEARCH RESULTS.

RULES:

1. Do not invent information.

2. Use only facts supported by the
   search results.

3. Mention relevant source names.

4. When possible, include source URLs.

5. If the search results do not contain
   enough reliable information, say:

   "I could not find enough reliable
   information from the web search."


WEB SEARCH RESULTS:
{web_context}


QUESTION:
{query}


ANSWER:
"""


    response = llm.invoke(
        prompt
    )


    return response.content


# ==================================================
# 14. REAL-TIME QUERY CHECK
# ==================================================

def needs_realtime_information(
    query
):

    query = query.lower()


    realtime_words = [

        "latest",

        "current",

        "today",

        "recent",

        "news",

        "live",

        "this week",

        "this month",

        "latest development",

        "latest developments",

        "new development",

        "new developments",

        "right now",

        "currently",

        "2026"
    ]


    return any(
        word in query
        for word in realtime_words
    )


# ==================================================
# 15. AGENTIC RAG ROUTER
# ==================================================

def ask_assistant(query):

    # ==================================================
    # CASE 1:
    # CURRENT / REAL-TIME INFORMATION
    # ==================================================

    if needs_realtime_information(
        query
    ):

        print(
            "\n[DECISION: CURRENT INFORMATION REQUIRED]"
        )

        print(
            "[ROUTING DIRECTLY TO WEB SEARCH]"
        )


        web_result = web_search(
            query
        )


        web_context = (
            format_web_results(
                web_result
            )
        )


        answer = answer_from_web(
            query,
            web_context
        )


        return f"""
==================================================
SOURCE: WEB SEARCH
==================================================

{answer}
""".strip()


    # ==================================================
    # CASE 2:
    # SEARCH LOCAL KNOWLEDGE FIRST
    # ==================================================

    context = vector_search(
        query
    )


    decision = check_context(
        query,
        context
    )


    print(
        "\nKnowledge Base Decision:",
        decision
    )


    # ==================================================
    # LOCAL KNOWLEDGE IS SUFFICIENT
    # ==================================================

    if decision == "YES":

        answer = answer_from_local_kb(
            query,
            context
        )


        return f"""
==================================================
SOURCE: LOCAL KNOWLEDGE BASE
==================================================

{answer}
""".strip()


    # ==================================================
    # LOCAL KNOWLEDGE INSUFFICIENT
    # WEB SEARCH FALLBACK
    # ==================================================

    else:

        print(
            "\n[DECISION: LOCAL KB INSUFFICIENT]"
        )

        print(
            "[FALLING BACK TO WEB SEARCH]"
        )


        web_result = web_search(
            query
        )


        web_context = (
            format_web_results(
                web_result
            )
        )


        answer = answer_from_web(
            query,
            web_context
        )


        return f"""
==================================================
SOURCE: WEB SEARCH
==================================================

{answer}
""".strip()


# ==================================================
# 16. APPLICATION HEADER
# ==================================================

print(
    "\n=============================================="
)

print(
    "        AGENTIC RESEARCH ASSISTANT"
)

print(
    "=============================================="
)


print(
    "\nFeatures:"
)

print(
    "- ChromaDB Local Knowledge Base"
)

print(
    "- HuggingFace Embeddings"
)

print(
    "- Semantic Similarity Search"
)

print(
    "- Knowledge Context Evaluation"
)

print(
    "- Tavily Web Search"
)

print(
    "- Automatic Web Search Fallback"
)

print(
    "- Real-Time Query Detection"
)

print(
    "- Agentic Decision Making"
)

print(
    "- Ollama Llama 3.2"
)


print(
    "\nAgentic Flow:"
)

print(
    "Retrieve -> Evaluate -> Decide -> Act -> Generate"
)


print(
    "\nType 'exit' to stop.\n"
)


# ==================================================
# 17. CHAT LOOP
# ==================================================

while True:

    user_input = input(
        "You: "
    )


    if (
        user_input.lower().strip()
        == "exit"
    ):

        print(
            "\nAgentic Research Assistant stopped."
        )

        break


    if not user_input.strip():

        print(
            "\nAssistant: Please enter a question.\n"
        )

        continue


    answer = ask_assistant(
        user_input
    )


    print(
        "\nAssistant:"
    )

    print(
        answer
    )

    print()