"""Opt-in end-to-end test against customer-owned Azure resources."""

from __future__ import annotations

import os

import pytest
from azure.identity import AzureCliCredential, get_bearer_token_provider
from openai import AzureOpenAI

from azure_openai_search_compat import OYDSearchChatClient
from tests.environment.config import required_environment, search_parameters

pytestmark = pytest.mark.environment

REQUIRED_VARIABLES = {
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_INDEX",
    "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
    "AZURE_SEARCH_CONTENT_FIELDS",
    "TEST_QUERY",
}


@pytest.mark.skipif(
    any(not os.environ.get(name) for name in REQUIRED_VARIABLES),
    reason="Live Azure environment variables are not configured.",
)
def test_grounded_answer_against_live_azure_resources() -> None:
    credential = AzureCliCredential()
    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default",
    )
    openai_client = AzureOpenAI(
        azure_endpoint=required_environment("AZURE_OPENAI_ENDPOINT"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_ad_token_provider=token_provider,
    )
    client = OYDSearchChatClient(openai_client, search_credential=credential)

    response = client.chat.completions.create(
        model=required_environment("AZURE_OPENAI_DEPLOYMENT"),
        messages=[
            {"role": "user", "content": required_environment("TEST_QUERY")}
        ],
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": search_parameters(),
                }
            ]
        },
        temperature=0,
    )

    message = response.choices[0].message
    assert message.content
    assert message.context["intent"]
    assert message.context["citations"]
    expected_text = os.environ.get("EXPECTED_TEXT")
    if expected_text:
        assert expected_text.casefold() in message.content.casefold()
