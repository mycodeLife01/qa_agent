from dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


@dataclass
class EmbeddingsConfig:
    model: str
    provider: str


@dataclass
class LLMConfig:
    model: str


@dataclass
class ModelConfig:
    embeddings: EmbeddingsConfig
    llm: LLMConfig


@dataclass
class FileConfig:
    allowed_types: list[str]


class SecretConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    huggingfacehub_api_token: str
    unstructured_api_key: str
    openai_api_key: str
    openai_api_base: str


@dataclass
class VDBConfig:
    persist_directory: str
    collection_name: str


@dataclass
class SystemConfig:
    model_config: ModelConfig
    file_config: FileConfig
    secret_config: SecretConfig
    vdb_config: VDBConfig


def load_config() -> SystemConfig:
    # 读取yaml配置文件
    with open("./config/config.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    embeddings_model = yaml_data["embeddings"]["model"]
    embeddings_provider = yaml_data["embeddings"]["provider"]
    allowed_types = yaml_data["file"]["allowed_types"]
    llm_model = yaml_data["llm"]["model"]
    vdb_persist_dir = yaml_data["vector_store"]["persist_directory"]
    vdb_collection_name = yaml_data["vector_store"]["collection_name"]
    model_config = ModelConfig(
        embeddings=EmbeddingsConfig(
            model=embeddings_model, provider=embeddings_provider
        ),
        llm=LLMConfig(model=llm_model),
    )
    file_config = FileConfig(allowed_types=allowed_types)
    vdb_config = VDBConfig(persist_directory=vdb_persist_dir, collection_name=vdb_collection_name)
    secret_config = SecretConfig()

    system_config = SystemConfig(
        model_config=model_config,
        file_config=file_config,
        secret_config=secret_config,
        vdb_config=vdb_config,
    )
    return system_config
