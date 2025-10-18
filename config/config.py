from dataclasses import dataclass
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


@dataclass
class EmbeddingsConfig:
    model: str
    provider: str


@dataclass
class ModelConfig:
    embeddings: EmbeddingsConfig


@dataclass
class FileConfig:
    allowed_types: list[str]


class SecretConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    huggingfacehub_api_token: str
    unstructured_api_key: str


@dataclass
class SystemConfig:
    model_config: ModelConfig
    file_config: FileConfig
    secret_config: SecretConfig


def load_config() -> SystemConfig:
    # 读取yaml配置文件
    with open("./config/config.yaml", "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    embeddings_model = yaml_data["embeddings"]["model"]
    embeddings_provider = yaml_data["embeddings"]["provider"]
    allowed_types = yaml_data["file"]["allowed_types"]

    model_config = ModelConfig(
        embeddings=EmbeddingsConfig(
            model=embeddings_model, provider=embeddings_provider
        )
    )
    file_config = FileConfig(allowed_types=allowed_types)
    secret_config = SecretConfig()

    system_config = SystemConfig(
        model_config=model_config, file_config=file_config, secret_config=secret_config
    )
    return system_config
