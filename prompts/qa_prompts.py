"""QA task prompt templates"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# QA 生成 Prompt - 使用 ChatPromptTemplate（推荐用于 Chat 模型）
QA_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.",
        ),
        (
            "human",
            "Question: {question}\n\nContext: {context}\n\nAnswer:",
        ),
    ]
)

# 如果需要使用简单的 PromptTemplate（用于非 Chat 模型）
QA_GENERATION_PROMPT_SIMPLE = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
)

