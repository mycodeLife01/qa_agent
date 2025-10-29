import time
from typing import TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from config import SystemConfig
from exception import (
    AgentMissingParamsException,
    AgentInvalidParamsException,
)
from langchain_chroma import Chroma
from prompts import QA_GENERATION_PROMPT
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEndpointEmbeddings


class QAAgent:
    class State(TypedDict):
        question: str
        context: list[Document]
        answer: str
        content_hash: str

    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.llm = ChatOpenAI(
            model=system_config.model_config.llm.model,
            api_key=system_config.secret_config.openai_api_key,
            base_url=system_config.secret_config.openai_api_base,
        )
        self.qa_prompt = QA_GENERATION_PROMPT
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=system_config.model_config.embeddings.model,
            provider=system_config.model_config.embeddings.provider,
            huggingfacehub_api_token=system_config.secret_config.huggingfacehub_api_token,
        )
        self.vector_store = Chroma(
            collection_name=system_config.vdb_config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.system_config.vdb_config.persist_directory,
        )
        self._init_graph()

    def _init_graph(self):
        graph_builder = StateGraph(self.State).add_sequence(
            [self.retrieve, self.generate]
        )
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    async def retrieve(self, state: State) -> State:
        # 查询问题
        retrieved_docs = self.vector_store.similarity_search(
            query=state["question"],
            filter={"content_hash": state["content_hash"]},
        )
        return {"context": retrieved_docs}

    async def generate(self, state: State) -> State:
        """使用 LLM 生成答案"""
        docs_content = "\n\n".join([doc.page_content for doc in state["context"]])

        # 使用 prompt 模板生成消息
        messages = self.qa_prompt.format_messages(
            question=state["question"], context=docs_content
        )

        # 调用 LLM 生成答案
        response = await self.llm.ainvoke(messages)

        return {"answer": response.content}

    async def run(self, state: State) -> str:
        params = list(state.keys())
        if len(params) == 0:
            raise AgentMissingParamsException
        if set(params) != set(["question", "content_hash"]):
            raise AgentInvalidParamsException

        return await self.graph.ainvoke(state)
