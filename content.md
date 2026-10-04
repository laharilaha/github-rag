================================================
FILE: README.md
================================================
# Agentic RAG using CrewAI

A local Agentic RAG application built with CrewAI, Qdrant, Chonkie, and Ollama.

Users can upload a PDF and ask questions about it. The application:
- extracts and chunks the document
- creates semantic embeddings
- stores vectors in an in-memory Qdrant database
- retrieves relevant content
- uses CrewAI agents to generate the final response
- runs locally with Ollama using `qwen2.5-coder:7b`

## Tech Stack

- Python
- CrewAI
- Qdrant
- Chonkie
- MarkItDown
- Ollama
- Qwen 2.5 Coder
- Streamlit


================================================
FILE: app.py
================================================
import os
import tempfile
import gc
import base64

import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM

from src.agentic_rag.tools.custom_tool import DocumentSearchTool


# =========================================================
# LLM
# =========================================================

@st.cache_resource
def load_llm():
    return LLM(
        model="ollama/qwen2.5-coder:7b",
        base_url="http://localhost:11434",
    )


# =========================================================
# CREATE CREW
# =========================================================

def create_crew(pdf_tool):
    llm = load_llm()

    retriever_agent = Agent(
        role="PDF Information Retriever",
        goal=(
            "Find the most relevant information from the uploaded PDF "
            "to answer the user's query."
        ),
        backstory=(
            "You are a careful document researcher. "
            "You must use the PDF search tool to find information "
            "before producing an answer."
        ),
        tools=[pdf_tool],
        llm=llm,
        verbose=True,
    )

    response_synthesizer_agent = Agent(
        role="Response Synthesizer",
        goal=(
            "Convert the retrieved PDF information into a clear, "
            "concise and accurate answer to the user's question."
        ),
        backstory=(
            "You are a precise technical communicator. "
            "Use the information retrieved from the document and "
            "do not invent facts."
        ),
        llm=llm,
        verbose=True,
    )

    retrieval_task = Task(
        description=(
            "Search the uploaded PDF for information relevant to "
            "this user query: {query}"
        ),
        expected_output=(
            "Relevant text retrieved from the uploaded PDF."
        ),
        agent=retriever_agent,
    )

    response_task = Task(
        description=(
            "Using the information retrieved by the previous task, "
            "answer this user query: {query}. "
            "If the document does not contain enough information, "
            "clearly say that the information was not found."
        ),
        expected_output=(
            "A concise and accurate answer based only on the "
            "retrieved document information."
        ),
        agent=response_synthesizer_agent,
    )

    crew = Crew(
        agents=[
            retriever_agent,
            response_synthesizer_agent,
        ],
        tasks=[
            retrieval_task,
            response_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "crew" not in st.session_state:
    st.session_state.crew = None


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="ðŸ“š",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("ðŸ“š Agentic RAG powered by CrewAI")
st.caption(
    "Chat with your PDF using semantic retrieval and a local Ollama LLM."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
    )

    if uploaded_file is not None:

        if st.session_state.pdf_tool is None:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:

                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            try:

                with st.spinner("Indexing PDF..."):

                    st.session_state.pdf_tool = DocumentSearchTool(
                        file_path=temp_file_path
                    )

                st.session_state.crew = None

                st.success("PDF indexed successfully!")

            except Exception as e:

                st.error("PDF indexing failed.")

                st.exception(e)

            finally:

                # The document has already been indexed into
                # the in-memory Qdrant database.
                try:
                    os.remove(temp_file_path)
                except OSError:
                    pass

    if st.button("Clear Chat"):

        st.session_state.messages = []
        gc.collect()

        st.rerun()


# =========================================================
# SHOW PDF STATUS
# =========================================================

if st.session_state.pdf_tool is None:

    st.info(
        "Upload a PDF from the sidebar to start chatting."
    )

else:

    st.success(
        f"Ready to answer questions about: {uploaded_file.name}"
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "Ask a question about your PDF..."
)


if prompt:

    if st.session_state.pdf_tool is None:

        st.warning(
            "Please upload a PDF before asking a question."
        )

        st.stop()


    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)


    # -----------------------------------------------------
    # CREATE CREW
    # -----------------------------------------------------

    if st.session_state.crew is None:

        with st.spinner("Preparing AI agents..."):

            st.session_state.crew = create_crew(
                st.session_state.pdf_tool
            )


    # -----------------------------------------------------
    # RUN CREW
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching the PDF and generating answer..."):

            try:

                result = st.session_state.crew.kickoff(
                    inputs={
                        "query": prompt
                    }
                )

                answer = result.raw

            except Exception as e:

                answer = (
                    "An error occurred while processing your question."
                )

                st.error(str(e))


        st.markdown(answer)


    # -----------------------------------------------------
    # SAVE RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


================================================
FILE: LICENSE
================================================
MIT License

Copyright (c) 2024 patchy631

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


================================================
FILE: pyproject.toml
================================================
[project]
name = "agentic_rag"
version = "0.1.0"
description = "agentic-rag using crewAI"
authors = [{ name = "Your Name", email = "you@example.com" }]
requires-python = ">=3.10,<=3.13"
dependencies = [
    "crewai[tools]>=0.86.0,<1.0.0"
]

[project.scripts]
agentic_rag = "agentic_rag.main:run"
run_crew = "agentic_rag.main:run"
train = "agentic_rag.main:train"
replay = "agentic_rag.main:replay"
test = "agentic_rag.main:test"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"



================================================
FILE: requirements.txt
================================================
crewai==0.203.2
streamlit
markitdown
chonkie[semantic]
qdrant-client


================================================
FILE: src/agentic_rag/__init__.py
================================================
[Empty file]


================================================
FILE: src/agentic_rag/crew.py
================================================
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from agentic_rag.tools.custom_tool import DocumentSearchTool


# PDF search tool
pdf_tool = DocumentSearchTool(
    file_path="knowledge/dspy.pdf"
)

# Web search tool
web_search_tool = SerperDevTool()


@CrewBase
class AgenticRag:
    """Agentic RAG crew."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["retriever_agent"],
            verbose=True,
            tools=[
                pdf_tool,
                web_search_tool
            ]
        )

    @agent
    def response_synthesizer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["response_synthesizer_agent"],
            verbose=True
        )

    @task
    def retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieval_task"],
        )

    @task
    def response_task(self) -> Task:
        return Task(
            config=self.tasks_config["response_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Agentic RAG crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


================================================
FILE: src/agentic_rag/main.py
================================================
#!/usr/bin/env python
import sys
import warnings

from agentic_rag.crew import AgenticRag

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'query': 'Who is elon musk?'
    }
    AgenticRag().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        AgenticRag().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AgenticRag().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        AgenticRag().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")



================================================
FILE: src/agentic_rag/config/agents.yaml
================================================
# routing_agent:
#   role: >
#     Understand the user query: {query}, and route it to the most appropriate knowledge base
#   goal: >
#     Route user queries to the most appropriate knowledge base
#   backstory: >
#     You are a user query routing agent. You are knowln for your ability to carefully understand user queries and route them to the most appropriate knowledge base.

retriever_agent:
  role: >
    Retrieve relevant information to answer the user query: {query}
  goal: >
    Retrieve the most relevant information from the available sources for the user query: {query}, always try to use the pdf search tool first. If you are not able to retrieve the information from the pdf search tool then try to use the web search tool.
  backstory: >
    You're a meticulous analyst with a keen eye for detail. You're known for
    your ability understand the user query: {query} and retrieve knowlege from the most suitable knowledge base.

response_synthesizer_agent:
  role: >
    Response synthesizer agent for the user query: {query}
  goal: >
    Synthesize the retrieved information into a concise and coherent response based on the user query: {query}. If you are not ble to retrieve the information then respond with "I'm sorry, I couldn't find the information you're looking for."
  backstory: >
    You're a skilled communicator with a knack for turning complex information into clear and concise responses.


================================================
FILE: src/agentic_rag/config/tasks.yaml
================================================
# routing_task:
#   description: >
#     Understand the user query: {query}, and route it to the most appropriate knowledge base
#   expected_output: >
#     The most appropriate knowledge base to route the user query to, the answer should be either 'pdf_search' or 'pdf_search'
#   agent: routing_agent

retrieval_task:
  description: >
    Retrieve the most relevant information from the available sources for the user query: {query}
  expected_output: >
    The most relevant information in form of text as retrieved from the sources.
  agent: retriever_agent

response_task:
  description: >
    Synthesize the final response for the user query: {query}
  expected_output: >
    A concise and coherent response based on the retrieved infromation from the right source for the user query: {query}. If you are not ble to retrieve the information then respond with "I'm sorry, I couldn't find the information you're looking for."
  agent: response_synthesizer_agent



================================================
FILE: src/agentic_rag/tools/__init__.py
================================================
[Empty file]


================================================
FILE: src/agentic_rag/tools/custom_tool.py
================================================
import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient, models


class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""

    query: str = Field(
        ...,
        description="Query to search the document.",
    )


class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the uploaded document for relevant information."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        super().__init__()

        self.file_path = file_path
        self.collection_name = "demo_collection"

        # In-memory Qdrant database
        self.client = QdrantClient(":memory:")

        # Embedding model
        self.embedding_model = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        ).embedding_model

        self._process_document()

    def _extract_text(self) -> str:
        """Extract text from PDF using MarkItDown."""

        md = MarkItDown()
        result = md.convert(self.file_path)

        return result.text_content

    def _create_chunks(self, raw_text: str):
        """Create semantic chunks from the document."""

        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        )

        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and store embeddings in Qdrant."""

        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)

        documents = [chunk.text for chunk in chunks]

        metadata = [
            {
                "source": os.path.basename(self.file_path)
            }
            for _ in chunks
        ]

        ids = list(range(len(documents)))

        # Create embeddings
        embeddings = self.embedding_model.embed_batch(documents)

        # Create Qdrant collection
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_model.dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        # Store vectors and document text
        points = [
            models.PointStruct(
                id=ids[i],
                vector=embeddings[i],
                payload={
                    "document": documents[i],
                    "metadata": metadata[i],
                },
            )
            for i in range(len(documents))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _run(self, query: str) -> str:
        """Search the document and return the most relevant chunks."""

        query_vector = self.embedding_model.embed(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=5,
        )

        docs = [
            f"Source: {point.payload['metadata']['source']}\n"
            f"{point.payload['document']}"
            for point in results.points
        ]

        return "\n___\n".join(docs)

