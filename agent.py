from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config.config import SystemConfig
from exception.qa_exception import UnsupportedFileTypeException
from utils.file_utils import process_file


class QAAgent:
    def __init__(self, system_config: SystemConfig):
        self.system_config = system_config
        self.allowed_types = system_config.file_config.allowed_types
        self.embeddings = system_config.model_config.embeddings.model
        self.embeddings_provider = system_config.model_config.embeddings.provider

    async def handle_qa(
        self, file_type: str, file_url: str, question: str
    ) -> str | None:
        # 处理源文件
        if file_type not in self.system_config.file_config.allowed_types:
            raise UnsupportedFileTypeException

        file = process_file(file_url)
        loader = UnstructuredLoader(file=file, partition_via_api=True)
        docs = loader.load()

        # 分割文件
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=20, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        # 生成向量
        embeddings = HuggingFaceEndpointEmbeddings(
            model=self.embeddings, provider=self.embeddings_provider
        )

        # 存储向量
        return
