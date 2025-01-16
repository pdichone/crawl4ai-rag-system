<!-- @format -->

\*\* must run pip install chromadb (or initially run the requirements.txt)

To Run:

First, you need to crawl and store the documentation:

python crawl_pydantic_ai_docs.py

This will crawl the Pydantic AI documentation and store it in your local ChromaDB.

Then:

Once the crawling is complete, you can run the Streamlit interface:

streamlit run ui.py
