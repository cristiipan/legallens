"""Centralized configuration loaded from environment variables.

We use pydantic-settings so config errors fail at startup (with helpful
messages) instead of at the first failed API call.
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Cohere — leave blank or set to "mock" to use the local sentence-
    # transformers fallback embedder + skip live LLM calls in tests.
    cohere_api_key: str = Field(default="")
    cohere_model: str = Field(default="command-r-plus")

    # Pinecone — same convention: empty/"mock" routes to LocalNumpyIndex.
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = Field(default="legallens-clauses")
    pinecone_environment: str = Field(default="us-east-1-aws")

    # Where the local fallback vector store persists its files.
    local_vector_root: str = Field(default="./data/vectors")

    # PostgreSQL
    database_url: str = Field(
        default="postgresql+asyncpg://legallens:legallens@localhost:5432/legallens"
    )

    # Redis Sentinel. `redis_sentinels` is a comma-separated `host:port` list.
    # The master_name must match the `sentinel monitor` directive in the
    # sentinel-*.conf files.
    redis_sentinels: str = Field(default="localhost:26379,localhost:26380,localhost:26381")
    redis_master_name: str = Field(default="legallens-master")

    # Logging
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Use this everywhere instead of instantiating Settings()."""
    return Settings()
