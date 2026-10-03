# RepoGuide AI

> Understand unfamiliar codebases with AI.

RepoGuide AI is a local AI assistant that helps developers understand unfamiliar software repositories.

Instead of manually searching through many files, a developer can provide a local repository, let RepoGuide AI index the relevant files, and ask questions about how the project works.

## Problem

Understanding an unfamiliar open-source codebase can take a lot of time.

Developers often need to:

- Find the important files
- Understand how components are connected
- Search through documentation and source code
- Figure out where a particular feature is implemented

RepoGuide AI provides a simple way to explore a repository using natural-language questions.

## How It Works

```text
Local Repository
       ↓
File Scanning
       ↓
Text Chunking
       ↓
ChromaDB
       ↓
Semantic Retrieval
       ↓
Llama 3.2 via Ollama
       ↓
Repository-specific Answer
       ↓
Relevant Source Files