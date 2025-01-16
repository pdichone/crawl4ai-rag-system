from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
from litellm import AsyncOpenAI
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client
from typing import List

import chromadb
from db import get_chroma_client, init_collection

load_dotenv()

llm = os.getenv("LLM_MODEL", "gpt-4o-mini")
model = OpenAIModel(llm)

logfire.configure(send_to_logfire="if-token-present")


@dataclass
class PydanticAIDeps:
    collection: chromadb.Collection  # ChromaDB collection
    openai_client: AsyncOpenAI


# Initialize ChromaDB collection
chroma_collection = init_collection()


system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_agent = Agent(
    model, system_prompt=system_prompt, deps_type=PydanticAIDeps, retries=2
)


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error


@pydantic_ai_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        # Query ChromaDB for relevant documents
        results = ctx.deps.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )

        if not results["documents"][0]:
            return "No relevant documentation found."

        # Format the results
        formatted_chunks = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunk_text = f"""
# {metadata['title']}

{doc}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@pydantic_ai_agent.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    """
    try:
        # Query ChromaDB for all documents
        results = ctx.deps.collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        # Extract unique URLs
        urls = sorted(set(meta["url"] for meta in results["metadatas"]))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@pydantic_ai_agent.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page.
    """
    try:
        # Query ChromaDB for all chunks of this URL
        results = ctx.deps.collection.get(
            where={"url": url}, include=["documents", "metadatas"]
        )

        if not results["documents"]:
            return f"No content found for URL: {url}"

        # Sort chunks by chunk_number
        sorted_results = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1]["chunk_number"],
        )

        # Format the page with its title and all chunks
        page_title = sorted_results[0][1]["title"].split(" - ")[0]
        formatted_content = [f"# {page_title}\n"]

        # Add each chunk's content
        for doc, _ in sorted_results:
            formatted_content.append(doc)

        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
