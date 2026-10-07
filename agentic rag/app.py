from langchain_groq import ChatGroq
from langchain.agents import create_agent

from tools import search_document, calculator


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# Tools
# --------------------------------------------------

tools = [
    search_document,
    calculator
]


# --------------------------------------------------
# Agent
# --------------------------------------------------

agent = create_agent(
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
"""
)


# --------------------------------------------------
# Chat Loop
# --------------------------------------------------

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
    )

    print("\nBot:", result["messages"][-1].content)