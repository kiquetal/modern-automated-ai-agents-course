"""
Streamlit app for Wikipedia QA with CrewAI and LangChain
- Parametrize model, chunking, and data sources
- Include optional RAG URLs and/or custom pasted text

Run: streamlit run app.py
"""
import os
import sys
import warnings
from typing import List

import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# Filter out specific warnings about USER_AGENT
warnings.filterwarnings("ignore", message="USER_AGENT environment variable not set")

# Load environment variables from .env if present
load_dotenv()

# Explicitly set USER_AGENT environment variable (helpful for website politeness and avoiding warnings)
os.environ["USER_AGENT"] = "WikipediaResearchBot/1.0 (Educational Research Bot)"

# Monkey patch sys.stderr to filter out specific warnings
_original_stderr_write = sys.stderr.write

def _filtered_stderr_write(text):
    if "USER_AGENT environment variable not set" not in text:
        _original_stderr_write(text)

sys.stderr.write = _filtered_stderr_write


def load_and_split_urls(urls: List[str], chunk_size: int, chunk_overlap: int):
    """Load web pages from URLs and split into chunks."""
    docs_all = []
    for url in urls:
        try:
            loader = WebBaseLoader(
                url,
                header_template={"User-Agent": "WikipediaResearchBot/1.0 (Educational Research Bot)"},
            )
            docs = loader.load()
            docs_all.extend(docs)
        except Exception as e:
            st.warning(f"Failed to load {url}: {e}")
    if not docs_all:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs_all)


def build_context_from_docs(docs, custom_text: str | None, source_list: List[str]) -> str:
    context_parts = []
    if source_list:
        context_parts.append("Sources:\n" + "\n".join(f"- {s}" for s in source_list))
    if docs:
        context_parts.append("Collected content chunks from sources:\n\n" + "\n\n".join(doc.page_content for doc in docs))
    if custom_text:
        context_parts.append("User-provided custom data:\n\n" + custom_text)
    return "\n\n".join(context_parts)


def build_questions_text(questions: List[str]) -> str:
    return "\n".join(f"- {q}" for q in questions if q.strip())


def run_crewai_flow(model_name: str, sources: List[str], questions: List[str], chunk_size: int, chunk_overlap: int, custom_text: str | None):
    """Create agents, tasks, context, and run CrewAI kickoff."""
    # Create LLM instance with selected model
    llm = ChatOpenAI(model=model_name)

    # Prepare documents and context
    documents = load_and_split_urls(sources, chunk_size, chunk_overlap)
    context = build_context_from_docs(documents, custom_text, sources)

    # Create agents
    researcher = Agent(
        role="Content Summarizer",
        goal="Summarize the provided text content",
        backstory=(
            "You are an expert at reading and summarizing text content to extract key information."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    analyst = Agent(
        role="Content Analyst",
        goal="Analyze content and provide insightful answers to questions",
        backstory=(
            "You understand complex information and provide clear, accurate answers. "
            "Ground answers strictly in the provided context."
        ),
        verbose=True,
        allow_delegation=True,
        llm=llm,
    )

    # Define tasks
    src_text = "\n".join(f"- {s}" for s in sources) if sources else "(no sources)"
    research_task = Task(
        description=(
            "Read and summarize the provided text content. The content was extracted from these sources:\n" + src_text
        ),
        expected_output="A comprehensive, well-structured summary of the provided content.",
        agent=researcher,
    )

    questions_text = build_questions_text(questions)
    analysis_task = Task(
        description=(
            "Based on the research summary and context, answer the following questions in detail, citing the relevant parts when possible:\n" + questions_text
        ),
        expected_output="Detailed answers to each question, grounded strictly in the provided content.",
        agent=analyst,
    )

    # Build crew
    crew = Crew(agents=[researcher, analyst], tasks=[research_task, analysis_task], verbose=True)

    # Provide context to research task
    research_task.context = context

    # Run
    result = crew.kickoff()
    return result


def parse_multiline(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Wikipedia QA (CrewAI)", page_icon="📚", layout="wide")
st.title("📚 Wikipedia and Custom Data QA (CrewAI + LangChain)")

with st.sidebar:
    st.header("Settings")
    # API Key handling
    default_key = os.getenv("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key", value=default_key, type="password", help="Will be set into environment for this session.")

    model = st.selectbox(
        "Model",
        options=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-3.5-turbo",
        ],
        index=0,
        help="Choose the OpenAI chat model.",
    )

    chunk_size = st.slider("Chunk size", min_value=500, max_value=4000, value=2000, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)

st.subheader("Sources")
wikipedia_urls_text = st.text_area(
    "Wikipedia URL(s)",
    value="https://en.wikipedia.org/wiki/Artificial_intelligence",
    help="Enter one URL per line.",
    height=100,
)

rag_urls_text = st.text_area(
    "Extra RAG URL(s) (optional)",
    value="",
    help="Add any additional URLs to include as custom data (one per line).",
    height=100,
)

custom_text = st.text_area(
    "Custom Text (optional)",
    value="",
    help="Paste any additional context you want the agents to use.",
    height=120,
)

st.subheader("Questions")
questions_text = st.text_area(
    "Questions (one per line)",
    value=(
        "What is the history of artificial intelligence?\n"
        "What are the main approaches to AI?\n"
        "What ethical concerns are associated with AI development?\n"
        "How is AI being used in industry today?"
    ),
    height=140,
)

run_button = st.button("Run QA")

if run_button:
    # Validate inputs
    if not api_key:
        st.error("Please provide your OpenAI API Key in the sidebar.")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key

    sources = parse_multiline(wikipedia_urls_text) + parse_multiline(rag_urls_text)
    questions = parse_multiline(questions_text)

    if not sources and not custom_text.strip():
        st.error("Please provide at least one source URL or some custom text.")
        st.stop()

    if not questions:
        st.error("Please provide at least one question.")
        st.stop()

    with st.spinner("Running agents... this may take a while"):
        try:
            result = run_crewai_flow(
                model_name=model,
                sources=sources,
                questions=questions,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                custom_text=custom_text.strip() or None,
            )
            st.success("Completed!")
        except Exception as e:
            st.error(f"Execution failed: {e}")
            result = None

    if result:
        st.subheader("Results")
        st.write(result)

st.markdown("""
---
Tip: You can set your OpenAI API key via a .env file (OPENAI_API_KEY=...) or in the sidebar field.
""")
