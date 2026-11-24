# Real-Time-Conversational-AI-Agentic-Decision-Assistant-
Developed a Real-Time Conversational AI & Agentic Decision Assistant designed to interact with users through natural language, understand their queries, and provide intelligent, context-aware responses. The project combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and agentic AI capabilities to create an interactive assistant that can retrieve relevant information, reason over user requests, and support decision-making.

The system uses Python as the core programming language and LangChain to manage LLM workflows, prompt engineering, conversation handling, and integration with different AI components. Ollama is used to run open-source language models locally, enabling private, cost-effective, and efficient AI interactions without relying entirely on cloud-based APIs.

To improve response accuracy and reduce hallucinations, FAISS (Facebook AI Similarity Search) is implemented as a vector database for storing and retrieving relevant information based on semantic similarity. This allows the assistant to search through knowledge sources, identify the most relevant context, and generate responses grounded in retrieved information.

The application is built using FastAPI, which provides a lightweight and high-performance backend for handling user requests, API communication, and real-time interaction between the user interface and AI processing components. The agentic functionality enables the assistant to analyze queries, determine appropriate actions, retrieve information, and generate meaningful recommendations rather than simply producing predefined responses.

Docker is used for containerization, ensuring consistent deployment and environment management across different systems. Git is utilized for version control, tracking code changes, and maintaining project development history.

Overall, this project demonstrates the practical implementation of conversational AI, local LLM deployment, semantic search, intelligent information retrieval, and agent-based decision support. It aims to deliver a scalable, modular, and responsive AI assistant capable of improving user experience through natural conversations, contextual understanding, and intelligent decision-making.
