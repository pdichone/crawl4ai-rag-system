from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
from datetime import datetime
import pytz

import streamlit as st
import json
import logfire
from openai import AsyncOpenAI
import chromadb

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter,
)
from pydantic_ai_agent import pydantic_ai_agent, PydanticAIDeps
from db import get_chroma_client, init_collection
from crawler import crawl_parallel, get_urls_from_sitemap

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB collection
chroma_collection = init_collection()

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire="never")


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str


def format_sitemap_url(url: str) -> str:
    """Format URL to ensure proper sitemap URL structure."""
    # Remove trailing slashes
    url = url.rstrip("/")

    # If URL doesn't end with sitemap.xml, append it
    if not url.endswith("sitemap.xml"):
        url = f"{url}/sitemap.xml"

    # Ensure proper URL structure
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    return url


def get_db_stats():
    """Get statistics and information about the current database."""
    try:
        # Get all documents and metadata
        results = chroma_collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return None

        # Get unique URLs
        urls = set(meta["url"] for meta in results["metadatas"])

        # Get domains/sources
        domains = set(meta["source"] for meta in results["metadatas"])

        # Get document count
        doc_count = len(results["ids"])

        # Format last updated time
        last_updated = max(meta.get("crawled_at", "") for meta in results["metadatas"])
        if last_updated:
            # Convert to local timezone
            dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.astimezone(local_tz)
            last_updated = dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        return {
            "urls": list(urls),
            "domains": list(domains),
            "doc_count": doc_count,
            "last_updated": last_updated,
        }
    except Exception as e:
        print(f"Error getting DB stats: {e}")
        return None


def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "urls_processed" not in st.session_state:
        st.session_state.urls_processed = set()
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "current_progress" not in st.session_state:
        st.session_state.current_progress = 0
    if "total_urls" not in st.session_state:
        st.session_state.total_urls = 0


def initialize_with_existing_data():
    """Check for existing data and initialize session state accordingly."""
    stats = get_db_stats()
    if stats and stats["doc_count"] > 0:
        st.session_state.processing_complete = True
        st.session_state.urls_processed = set(stats["urls"])
        return stats
    return None


def display_message_part(part):
    """Display a single part of a message in the Streamlit UI."""
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str):
    """Run the agent with streaming text for the user_input prompt."""
    deps = PydanticAIDeps(collection=chroma_collection, openai_client=openai_client)

    async with pydantic_ai_agent.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        filtered_messages = [
            msg
            for msg in result.new_messages()
            if not (
                hasattr(msg, "parts")
                and any(part.part_kind == "user-prompt" for part in msg.parts)
            )
        ]
        st.session_state.messages.extend(filtered_messages)

        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def process_url(url: str):
    """Process a single URL or sitemap URL."""
    try:
        progress_container = st.empty()
        with progress_container.container():
            # Format the URL
            formatted_url = format_sitemap_url(url)
            st.write(f"🔄 Processing {formatted_url}...")

            # First try as sitemap
            st.write("📑 Attempting to fetch sitemap...")
            urls = get_urls_from_sitemap(formatted_url)

            if urls:
                st.write(f"📎 Found {len(urls)} URLs in sitemap")
                # Create a progress bar
                progress_bar = st.progress(0, text="Processing URLs...")
                st.session_state.total_urls = len(urls)

                # Process URLs
                await crawl_parallel(urls)
                progress_bar.progress(100, text="Processing complete!")
            else:
                # If sitemap fails, try processing as single URL
                st.write("❌ No sitemap found or empty sitemap.")
                st.write("🔍 Attempting to process as single URL...")
                original_url = url.rstrip(
                    "/sitemap.xml"
                )  # Remove sitemap.xml if it was added
                st.session_state.total_urls = 1
                await crawl_parallel([original_url])

            # Show summary of processed documents
            try:
                doc_count = len(chroma_collection.get()["ids"])
                st.success(
                    f"✅ Processing complete! Total documents in database: {doc_count}"
                )
            except Exception as e:
                st.error(f"Unable to get document count: {str(e)}")

    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")


async def main():
    st.set_page_config(
        page_title="Dynamic RAG Chat System", page_icon="🤖", layout="wide"
    )

    initialize_session_state()

    # Check for existing data
    existing_data = initialize_with_existing_data()

    st.title("Dynamic RAG Chat System")

    # Show system status and information
    if existing_data:
        st.success("💡 System is ready with existing knowledge base!")
        with st.expander("Knowledge Base Information", expanded=True):
            st.markdown(
                f"""
            ### Current Knowledge Base Stats:
            - 📚 Number of documents: {existing_data['doc_count']}
            - 🌐 Number of sources: {len(existing_data['domains'])}
            - 🕒 Last updated: {existing_data['last_updated']}
            
            ### Sources include:
            {', '.join(existing_data['domains'])}
            
            ### You can ask questions about:
            - Any content from the processed websites
            - Specific information from any of the loaded pages
            - Technical details, documentation, or other content from these sources
            
            ### Loaded URLs:
            """
            )
            for url in existing_data["urls"]:
                st.write(f"- {url}")
    else:
        st.info("👋 Welcome! Start by adding a website to create your knowledge base.")

    # Create two main columns for the layout
    input_col, chat_col = st.columns([1, 2])

    with input_col:
        st.subheader("Add Content to RAG System")
        st.write("Enter a website URL to process. The system will:")
        st.write(
            "1. First try to find and process the sitemap (automatically appending '/sitemap.xml')"
        )
        st.write("2. If no sitemap is found, process the URL as a single page")

        url_input = st.text_input(
            "Website URL",
            key="url_input",
            placeholder="example.com or https://example.com",
        )

        # Show the formatted URL preview if input exists
        if url_input:
            formatted_preview = format_sitemap_url(url_input)
            st.caption(f"Will try: {formatted_preview}")

        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button(
                "Process URL", disabled=st.session_state.is_processing, type="primary"
            )
        with col2:
            if st.button(
                "Clear Database",
                disabled=st.session_state.is_processing,
                type="secondary",
            ):
                try:
                    # Get all document IDs
                    all_ids = chroma_collection.get()["ids"]
                    if all_ids:  # Only attempt to delete if there are documents
                        chroma_collection.delete(ids=all_ids)
                    # Reset session state
                    st.session_state.processing_complete = False
                    st.session_state.urls_processed = set()
                    st.session_state.messages = []
                    st.success("Database cleared successfully!")
                    # Force page refresh to update stats
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing database: {str(e)}")

        if process_button and url_input:
            if url_input not in st.session_state.urls_processed:
                st.session_state.is_processing = True
                await process_url(url_input)
                st.session_state.urls_processed.add(url_input)
                st.session_state.processing_complete = True
                st.session_state.is_processing = False
                st.rerun()  # Refresh to update stats
            else:
                st.warning("This URL has already been processed!")

        # Display processed URLs
        if st.session_state.urls_processed:
            st.subheader("Processed URLs:")
            for processed_url in st.session_state.urls_processed:
                st.write(f"✓ {processed_url}")

    with chat_col:
        if st.session_state.processing_complete:
            st.subheader("Chat Interface")

            # Add suggested questions based on content
            with st.expander("📝 Suggested Questions", expanded=False):
                st.markdown(
                    """
                Try asking:
                - "What topics are covered in the knowledge base?"
                - "Can you summarize the main content from [specific domain]?"
                - "What are the key concepts discussed across these documents?"
                - "Find information about [specific topic] from these sources"
                """
                )

            # Display existing messages
            for msg in st.session_state.messages:
                if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                    for part in msg.parts:
                        display_message_part(part)

            # Chat input
            user_input = st.chat_input(
                "Ask a question about the processed content...",
                disabled=st.session_state.is_processing,
            )

            if user_input:
                st.session_state.messages.append(
                    ModelRequest(parts=[UserPromptPart(content=user_input)])
                )

                with st.chat_message("user"):
                    st.markdown(user_input)

                with st.chat_message("assistant"):
                    await run_agent_with_streaming(user_input)

            # Clear chat button
            if st.button("Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        else:
            if existing_data:
                st.info("The knowledge base is ready! Start asking questions below.")
            else:
                st.info("Please process a URL first to start chatting!")

    # Add footer with system status
    st.markdown("---")
    if existing_data:
        st.markdown(
            f"System Status: 🟢 Ready with {existing_data['doc_count']} documents from {len(existing_data['domains'])} sources"
        )
    else:
        st.markdown("System Status: 🟡 Waiting for content")


if __name__ == "__main__":
    asyncio.run(main())
