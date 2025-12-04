import time
from typing import TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from config import SystemConfig
from exception import (
    AgentMissingParamsException,
    AgentInvalidParamsException,
)
import chromadb
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
        # 使用 ChromaDB 服务器模式（HttpClient）
        self.chroma_client = chromadb.HttpClient(
            host=system_config.vdb_config.chroma_host,
            port=system_config.vdb_config.chroma_port,
        )
        self.vector_store = Chroma(
            collection_name=system_config.vdb_config.collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client,
        )
        self._init_graph()
    
    def _recreate_vector_store(self):
        """重新创建vector_store实例，用于处理连接错误"""
        print("[Agent] 重新创建vector_store实例...")
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model=self.system_config.model_config.embeddings.model,
            provider=self.system_config.model_config.embeddings.provider,
            huggingfacehub_api_token=self.system_config.secret_config.huggingfacehub_api_token,
        )
        # 重新创建 HttpClient 连接
        self.chroma_client = chromadb.HttpClient(
            host=self.system_config.vdb_config.chroma_host,
            port=self.system_config.vdb_config.chroma_port,
        )
        self.vector_store = Chroma(
            collection_name=self.system_config.vdb_config.collection_name,
            embedding_function=self.embeddings,
            client=self.chroma_client,
        )

    def _init_graph(self):
        graph_builder = StateGraph(self.State).add_sequence(
            [self.retrieve, self.generate]
        )
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    async def retrieve(self, state: State) -> State:
        # 查询问题 - 如果遇到连接错误则重试
        print(f"[Agent] 开始检索 - 问题: {state['question']}, content_hash: {state['content_hash']}")
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                retrieved_docs = self.vector_store.similarity_search(
                    query=state["question"],
                    filter={"content_hash": state["content_hash"]},
                )
                print(f"[Agent] 检索成功 - 找到 {len(retrieved_docs)} 个文档")
                return {"context": retrieved_docs}
            except Exception as e:
                last_error = e
                error_str = str(e)
                # 检查是否是SSL/连接错误
                if "SSL" in error_str or "EOF" in error_str or "Connection" in error_str:
                    print(f"[Agent] 检测到连接错误 (尝试 {attempt + 1}/{max_retries}): {error_str}")
                    if attempt < max_retries - 1:
                        # 重新创建vector_store实例
                        self._recreate_vector_store()
                        continue
                # 其他错误直接抛出
                raise
        
        # 所有重试都失败
        raise last_error

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
