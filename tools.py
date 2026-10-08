from langchain.tools import tool

from rag import retriever


@tool
def search_document(query: str) -> str:
    """
    Search the uploaded PDF and return relevant information.
    Use this tool when the question is related to the document.
    """

    docs = retriever.invoke(query)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:
        return str(eval(expression))
    except:
        return "Invalid mathematical expression."