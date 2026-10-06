"""Environment configuration shared by live and evaluation tests."""

from __future__ import annotations

import os
from typing import Any


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable.")
    return value


def comma_separated_environment(name: str) -> list[str]:
    return [
        value.strip()
        for value in os.environ.get(name, "").split(",")
        if value.strip()
    ]


def search_parameters() -> dict[str, Any]:
    authentication: dict[str, Any]
    api_key = os.environ.get("AZURE_SEARCH_API_KEY")
    if api_key:
        authentication = {"type": "api_key", "key": api_key}
    else:
        authentication = {"type": "system_assigned_managed_identity"}

    field_mapping: dict[str, Any] = {
        "content_fields": comma_separated_environment(
            "AZURE_SEARCH_CONTENT_FIELDS"
        ),
        "vector_fields": comma_separated_environment(
            "AZURE_SEARCH_VECTOR_FIELDS"
        ),
    }
    optional_fields = {
        "title_field": os.environ.get("AZURE_SEARCH_TITLE_FIELD"),
        "url_field": os.environ.get("AZURE_SEARCH_URL_FIELD"),
        "filepath_field": os.environ.get("AZURE_SEARCH_FILEPATH_FIELD"),
    }
    field_mapping.update(
        {name: value for name, value in optional_fields.items() if value}
    )

    return {
        "endpoint": required_environment("AZURE_SEARCH_ENDPOINT"),
        "index_name": required_environment("AZURE_SEARCH_INDEX"),
        "authentication": authentication,
        "query_type": os.environ.get("AZURE_SEARCH_QUERY_TYPE", "semantic"),
        "semantic_configuration": os.environ.get(
            "AZURE_SEARCH_SEMANTIC_CONFIGURATION"
        ),
        "fields_mapping": field_mapping,
        "filter": os.environ.get("AZURE_SEARCH_FILTER"),
        "strictness": int(os.environ.get("AZURE_SEARCH_STRICTNESS", "3")),
        "top_n_documents": int(
            os.environ.get("AZURE_SEARCH_TOP_N_DOCUMENTS", "5")
        ),
        "in_scope": os.environ.get("AZURE_SEARCH_IN_SCOPE", "true").lower()
        == "true",
    }
