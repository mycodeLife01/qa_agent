import time
from typing import TypedDict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config import SystemConfig
from exception import (
    AgentUnsupportedFileTypeException,
    AgentMissingParamsException,
    AgentInvalidParamsException,
)
from utils import process_file
from langchain_chroma import Chroma
from prompts import QA_GENERATION_PROMPT
from langgraph.graph import START, StateGraph
from langchain_community.vectorstores.utils import filter_complex_metadata

class QAAgent:
    class State(TypedDict):
        question: str
        context: list[Document]
        answer: str
        file_type: str
        file_url: str

    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.allowed_types = system_config.file_config.allowed_types
        self.embeddings = system_config.model_config.embeddings.model
        self.embeddings_provider = system_config.model_config.embeddings.provider
        self.llm = ChatOpenAI(
            model=system_config.model_config.llm.model,
            api_key=system_config.secret_config.openai_api_key,
            base_url=system_config.secret_config.openai_api_base,
        )
        self.qa_prompt = QA_GENERATION_PROMPT
        self._init_graph()

    def _init_graph(self):
        graph_builder = StateGraph(self.State).add_sequence(
            [self.handle_qa, self.generate]
        )
        graph_builder.add_edge(START, "handle_qa")
        self.graph = graph_builder.compile()

    async def handle_qa(self, state: State) -> State:
        # 处理源文件
        if state["file_type"] not in self.system_config.file_config.allowed_types:
            raise AgentUnsupportedFileTypeException

        file = await process_file(state["file_url"])

        start_partition_time = time.time()
        loader = UnstructuredLoader(
            file=file,
            metadata_filename=state["file_url"],
            # api_key=self.system_config.secret_config.unstructured_api_key,
            # partition_via_api=True,
        )
        docs = loader.load()
        end_partition_time = time.time()
        print(
            f"[DEBUG] Partition time: {end_partition_time - start_partition_time} seconds"
        )

        # 分割文件
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)
        filtered_splits = filter_complex_metadata(all_splits)

        # 生成并存储向量
        embeddings = HuggingFaceEndpointEmbeddings(
            model=self.embeddings,
            provider=self.embeddings_provider,
            huggingfacehub_api_token=self.system_config.secret_config.huggingfacehub_api_token
        )
        vector_store = Chroma(
            collection_name="qa_vector_store",
            embedding_function=embeddings,
            persist_directory="./vector_store",
        )
         # 检查数据库是否为空，避免重复添加
        if vector_store._collection.count() == 0:
            _ = vector_store.add_documents(filtered_splits)
            print(f"[INFO]: Added {len(filtered_splits)} documents to vector store")
        else:
            print(f"[INFO]: Vector store already has {vector_store._collection.count()} documents, skipping add")

        # 查询问题
        retrieved_docs = vector_store.similarity_search(state["question"])
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
        if set(params) != set(["file_type", "file_url", "question"]):
            raise AgentInvalidParamsException

        return await self.graph.ainvoke(state)
