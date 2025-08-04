# Agentic AI - Document QI and Router

An LLM-powered, multi-agent pipeline that ingests documents and answers user questions by routing each query to the best tool: 
summarization, semantic search, or action/escape. The system supports both high-level overviews and fine-grained Q&A, 
and can escalate out-of-scope requests to external tools/APIs.



## Technical Report
https://discovered-duck-af0.notion.site/Task-3-Autonomous-Document-QA-Routing-244066d9a43380b4b8c7da105af03079


## Modules
- pipeline.py	=> loads document and query, and decides which agent to route. It uses llama_index, and HuggingFaceEmbedding to do the job.
- agents.py	=> defines agents.
    - summary agent: provides high-level document overview
    - search agent: performs vector-based semantic retrieval
    - action agent: handles out-of-scope queries (currently a placeholder, and later extendable to external APIs)
- embeddingSelector.py => implements EmbeddingAwareSelector, a custom router that enhances default LLM-based routing by incorporating vector similarity scores and the top-matching content snippet from the document. Improves decision-making by giving the LLM more context.
- build_graph.py => assembles the LangGraph agent workflows, wiring together the summary, search, action, and merge agents.


## Installation
1. Clone the git
2. (Optional) Create a virtual environment
   - python -m venv venv
   - source venv/bin/activate  # or venv\Scripts\activate on Windows
3. Install dependencies
   - pip install -r requirements.txt
4. Add OpenAI API key
   - make .env file and add key
   - OPENAI_API_KEY=your_key
4. Run the file
   - python app/main.py
   - It will run the test with test_cases queries in main.py and a given document in /docs.
