"""
Model registry for dynamic model instantiation.
"""

from typing import Type

from .base import BaseModel
from .mlp import MLPModel

# Global model registry
_MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "mlp": MLPModel,
}


def register_model(name: str, model_class: Type[BaseModel]):
    """
    Register a new model class.

    Args:
        name: Model name for config
        model_class: Model class to register
    """
    _MODEL_REGISTRY[name] = model_class


def get_model(name: str, **kwargs) -> BaseModel:
    """
    Instantiate a model by name.

    Args:
        name: Model name (mlp, etc.)
        **kwargs: Model constructor arguments

    Returns:
        Model instance
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. " f"Available: {list(_MODEL_REGISTRY.keys())}"
        )

    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return list of available model names."""
    return list(_MODEL_REGISTRY.keys())
