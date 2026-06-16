from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InMemoryConfig(BaseModel):
    """In-memory vector store config for benchmark-friendly upstream Mem0.

    This adapter avoids external services (e.g., qdrant) while keeping Mem0's
    algorithmic flow (extract -> update -> CRUD) intact.
    """

    collection_name: str = Field("mem0", description="Collection name")
    distance_strategy: str = Field(
        "cosine", description="Distance strategy. Options: 'cosine' or 'inner_product'"
    )
    embedding_model_dims: int = Field(1536, description="Dimension of the embedding vector")

    @model_validator(mode="before")
    @classmethod
    def validate_distance_strategy(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        ds = values.get("distance_strategy")
        if ds and ds not in ["cosine", "inner_product"]:
            raise ValueError("Invalid distance_strategy. Must be one of: 'cosine', 'inner_product'")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)
