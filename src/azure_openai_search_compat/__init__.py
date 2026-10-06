"""Compatibility client for replacing Azure OpenAI On Your Data."""

from azure_openai_search_compat.client import (
    ConfigurationError,
    OYDSearchChatClient,
    UnsupportedConfigurationError,
)

__all__ = [
    "ConfigurationError",
    "OYDSearchChatClient",
    "UnsupportedConfigurationError",
]

__version__ = "0.1.0"
