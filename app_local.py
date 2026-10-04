import os
import gc
import tempfile
import uuid
import pandas as pd

from gitingest import ingest

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import MarkdownNodeParser

import streamlit as st


# -----------------------------
# Session State
# -----------------------------

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id


# -----------------------------
# Load Local LLM
# -----------------------------

@st.cache_resource
def load_llm():
    llm = Ollama(
        model="qwen2.5-coder:7b",
        request_timeout=300.0
    )
    return llm


# -----------------------------
# Reset Chat
# -----------------------------

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


# -----------------------------
# GitHub Repository Processing
# -----------------------------

def process_with_gitingest(github_url):
    summary, tree, content = ingest(github_url)
    return summary, tree, content


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("Add your GitHub repository!")

    github_url = st.text_input(
        "Enter GitHub repository URL",
        placeholder="GitHub URL"
    )

    load_repo = st.button("Load Repository")


    if github_url and load_repo:

        try:

            with tempfile.TemporaryDirectory() as temp_dir:

                st.write("Processing your repository...")

                repo_name = github_url.split("/")[-1]

                file_key = f"{session_id}-{repo_name}"


                if file_key not in st.session_state.file_cache:

                    # Process GitHub repository
                    summary, tree, content = process_with_gitingest(
                        github_url
                    )


                    # Save repository content
                    content_path = os.path.join(
                        temp_dir,
                        f"{repo_name}_content.md"
                    )

                    with open(
                        content_path,
                        "w",
                        encoding="utf-8"
                    ) as f:

                        f.write(content)


                    # Load markdown document
                    loader = SimpleDirectoryReader(
                        input_dir=temp_dir
                    )

                    docs = loader.load_data()


                    # -----------------------------
                    # Embedding Model
                    # -----------------------------

                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-large-en-v1.5",
                        trust_remote_code=True
                    )

                    Settings.embed_model = embed_model


                    # -----------------------------
                    # Markdown Parser
                    # -----------------------------

                    node_parser = MarkdownNodeParser()


                    # -----------------------------
                    # Create Vector Index
                    # -----------------------------

                    index = VectorStoreIndex.from_documents(
                        documents=docs,
                        transformations=[node_parser],
                        show_progress=True
                    )


                    # -----------------------------
                    # Local LLM
                    # -----------------------------

                    llm = load_llm()

                    Settings.llm = llm


                    # -----------------------------
                    # Query Engine
                    # -----------------------------

                    query_engine = index.as_query_engine(
                        streaming=True
                    )


                    # -----------------------------
                    # Custom Prompt
                    # -----------------------------

                    qa_prompt_tmpl_str = (

                        "Context information is below.\n"

                        "---------------------\n"

                        "{context_str}\n"

                        "---------------------\n"

                        "Given the context information above, "
                        "answer the query clearly and precisely. "

                        "Focus only on the GitHub repository content. "

                        "If you don't know the answer, say "
                        "'I don't know!'.\n"

                        "Query: {query_str}\n"

                        "Answer: "
                    )


                    qa_prompt_tmpl = PromptTemplate(
                        qa_prompt_tmpl_str
                    )


                    query_engine.update_prompts(
                        {
                            "response_synthesizer:text_qa_template":
                            qa_prompt_tmpl
                        }
                    )


                    # Cache query engine
                    st.session_state.file_cache[file_key] = query_engine

                else:

                    query_engine = (
                        st.session_state.file_cache[file_key]
                    )


                st.success("Ready to Chat!")


        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )

            st.stop()


# -----------------------------
# Main Page
# -----------------------------

col1, col2 = st.columns([6, 1])


with col1:

    st.header(
        "Chat with GitHub using RAG </>"
    )


with col2:

    st.button(
        "Clear ↺",
        on_click=reset_chat
    )


# -----------------------------
# Initialize Chat History
# -----------------------------

if "messages" not in st.session_state:

    reset_chat()


# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# -----------------------------
# User Input
# -----------------------------

if prompt := st.chat_input("What's up?"):

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )


    # Display user message
    with st.chat_message("user"):

        st.markdown(prompt)


    # Assistant response
    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        full_response = ""


        try:

            # Get repository name
            repo_name = github_url.split("/")[-1]

            file_key = (
                f"{session_id}-{repo_name}"
            )


            # Get query engine
            query_engine = (
                st.session_state.file_cache.get(
                    file_key
                )
            )


            if query_engine is None:

                st.error(
                    "Please load a repository first!"
                )

                st.stop()


            # Query the RAG system
            response = query_engine.query(
                prompt
            )


            # Streaming response
            if hasattr(
                response,
                "response_gen"
            ):

                for chunk in response.response_gen:

                    if isinstance(chunk, str):

                        full_response += chunk

                        message_placeholder.markdown(
                            full_response + "▌"
                        )


            else:

                full_response = str(response)

                message_placeholder.markdown(
                    full_response
                )


            # Final response
            message_placeholder.markdown(
                full_response
            )


        except Exception as e:

            st.error(
                f"An error occurred while processing your query: {str(e)}"
            )

            full_response = (
                "Sorry, I encountered an error "
                "while processing your request."
            )

            message_placeholder.markdown(
                full_response
            )


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )