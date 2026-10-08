import glob
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from tools import calculator, search_document

load_dotenv()


def build_agent():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    tools = [
        search_document,
        calculator,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
You are an intelligent assistant for multiple PDF documents.

You have access to two tools:

1. search_document
   - Search across all uploaded PDF documents.
   - ALWAYS use this tool when the user's question
     requires information from the uploaded documents.
   - Do not answer document-related questions using
     your general knowledge.
   - Use only information returned by the document search.
   - If the requested information is not found in the
     uploaded documents, clearly say that it was not found.
   - When possible, mention the document name and page
     that support the answer.

2. calculator
   - Use this tool for mathematical calculations.
   - Do not perform calculations yourself when the
     calculator tool can be used.

Do not invent information from the documents.
""",
    )


def streamlit_app():
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Install streamlit to use the web UI: pip install streamlit") from exc

    st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="📚")
    st.title("Agentic RAG Chatbot")
    st.caption("Ask questions from your PDF knowledge base.")

    pdf_count = len(glob.glob("documents/*.pdf"))
    with st.sidebar:
        st.header("Knowledge base")
        st.metric("PDF documents", pdf_count)
        st.write("Source folder: documents/")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Cache agent so it's not rebuilt on every Streamlit rerun
    @st.cache_resource
    def get_agent():
        return build_agent()

    agent = get_agent()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about the uploaded PDFs...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Searching the documents..."):
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })

        answer = result["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


def cli_loop():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    agent = build_agent()

    while True:
        question = input("\nYou: ")

        if question.lower() == "exit":
            break

        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            }
        )

        print("\nBot:", result["messages"][-1].content)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "cli":
        cli_loop()
    else:
        streamlit_app()