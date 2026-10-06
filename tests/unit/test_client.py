"""Tests for the OYD-compatible Azure AI Search chat client."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from azure_openai_search_compat import (
    ConfigurationError,
    OYDSearchChatClient,
    UnsupportedConfigurationError,
)


def _openai_client(
    intents: list[str] | None = None,
) -> tuple[MagicMock, SimpleNamespace]:
    message = SimpleNamespace(content="Grounded answer", role="assistant")
    response = SimpleNamespace(
        id="completion-1",
        choices=[SimpleNamespace(index=0, message=message, finish_reason="stop")],
    )
    client = MagicMock()
    if intents is None:
        client.chat.completions.create.return_value = response
    else:
        intent_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=__import__("json").dumps({"intents": intents})
                    )
                )
            ]
        )
        client.chat.completions.create.side_effect = [intent_response, response]
    return client, response


def _data_source(**parameter_overrides):
    parameters = {
        "endpoint": "https://example.search.windows.net",
        "index_name": "customer-index",
        "authentication": {"type": "api_key", "key": "search-key"},
        "query_type": "semantic",
        "semantic_configuration": "default",
        "fields_mapping": {
            "content_fields": ["content"],
            "title_field": "title",
            "url_field": "url",
            "vector_fields": ["contentVector"],
        },
        "strictness": 3,
        "top_n_documents": 2,
        "in_scope": True,
    }
    parameters.update(parameter_overrides)
    return {"type": "azure_search", "parameters": parameters}


@patch("azure_openai_search_compat.client.SearchClient")
def test_preserves_call_shape_and_adds_citations(search_client_type):
    search_client_type.return_value.search.return_value = [
        {
            "content": "The customer policy allows 20 vacation days.",
            "title": "Benefits",
            "url": "https://example.test/benefits",
            "@search.reranker_score": 3.5,
        }
    ]
    openai_client, response = _openai_client(["employee vacation policy"])
    client = OYDSearchChatClient(openai_client)

    result = client.chat.completions.create(
        model="gpt-4o-canada",
        messages=[{"role": "user", "content": "What is the vacation policy?"}],
        extra_body={"data_sources": [_data_source()]},
        temperature=0.2,
    )

    assert result.id == response.id
    assert result.choices[0].message.content == "Grounded answer"
    assert result.choices[0].message.context["citations"][0]["title"] == "Benefits"
    search_client_type.return_value.search.assert_called_once_with(
        search_text="employee vacation policy",
        top=4,
        select=["content", "title", "url"],
        query_type="semantic",
        semantic_configuration_name="default",
    )
    search_client_type.return_value.close.assert_called_once_with()

    assert result.choices[0].message.context["intent"] == [
        "employee vacation policy"
    ]
    assert openai_client.chat.completions.create.call_count == 2
    intent_call = openai_client.chat.completions.create.call_args_list[0].kwargs
    assert intent_call["response_format"] == {"type": "json_object"}
    call = openai_client.chat.completions.create.call_args_list[1].kwargs
    assert call["model"] == "gpt-4o-canada"
    assert call["temperature"] == 0.2
    assert call["extra_body"] is None
    assert "data_sources" not in call
    assert "Retrieved documents:" in call["messages"][0]["content"]
    assert "Title: Benefits" in call["messages"][0]["content"]
    assert "20 vacation days" in call["messages"][0]["content"]


def test_without_data_sources_delegates_unchanged():
    openai_client, response = _openai_client()
    messages = [{"role": "user", "content": "Hello"}]
    client = OYDSearchChatClient(openai_client)

    result = client.chat.completions.create(model="gpt-4o", messages=messages)

    assert result is response
    openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=messages,
        extra_body=None,
    )


@patch("azure_openai_search_compat.client.SearchClient")
def test_vector_hybrid_uses_search_vectorizer(search_client_type):
    search_client_type.return_value.search.return_value = []
    openai_client, _ = _openai_client(["company policy"])
    client = OYDSearchChatClient(openai_client)

    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Find the policy"}],
        extra_body={
            "data_sources": [
                _data_source(
                    query_type="vector_simple_hybrid",
                    semantic_configuration=None,
                )
            ]
        },
    )

    search_call = search_client_type.return_value.search.call_args.kwargs
    assert search_call["search_text"] == "company policy"
    assert search_call["vector_queries"][0].text == "company policy"
    assert search_call["vector_queries"][0].fields == "contentVector"


def test_streaming_fails_explicitly():
    openai_client, _ = _openai_client()
    client = OYDSearchChatClient(openai_client)

    with pytest.raises(UnsupportedConfigurationError, match="streaming"):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )


def test_semantic_query_requires_configuration():
    openai_client, _ = _openai_client(["hello"])
    client = OYDSearchChatClient(openai_client)

    with pytest.raises(ConfigurationError, match="semantic_configuration"):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={
                "data_sources": [_data_source(semantic_configuration=None)]
            },
        )


@patch("azure_openai_search_compat.client.SearchClient")
def test_multiple_intents_are_searched_and_duplicate_results_are_fused(search_client_type):
    duplicate = {
        "content": "Twenty days of annual leave.",
        "title": "Leave policy",
        "url": "https://example.test/leave",
        "@search.reranker_score": 4.0,
    }
    search_client_type.return_value.search.side_effect = [
        [duplicate],
        [duplicate],
    ]
    openai_client, _ = _openai_client(
        ["annual leave allowance", "vacation days per year"]
    )
    client = OYDSearchChatClient(openai_client)

    result = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "Tell me about employee benefits."},
            {"role": "assistant", "content": "Which benefit?"},
            {"role": "user", "content": "How many days do I get?"},
        ],
        extra_body={"data_sources": [_data_source()]},
    )

    assert search_client_type.return_value.search.call_count == 2
    assert len(result.choices[0].message.context["citations"]) == 1
    assert result.choices[0].message.context["intent"] == [
        "annual leave allowance",
        "vacation days per year",
    ]


@patch("azure_openai_search_compat.client.SearchClient")
def test_serialized_response_contains_oyd_context(search_client_type):
    search_client_type.return_value.search.return_value = [
        {
            "content": "The customer policy allows 20 vacation days.",
            "title": "Benefits",
            "url": "https://example.test/benefits",
            "@search.reranker_score": 4.0,
        }
    ]
    openai_client = MagicMock()
    intent_message = MagicMock()
    intent_message.content = '{"intents":["employee vacation policy"]}'
    intent_response = MagicMock()
    intent_response.choices = [SimpleNamespace(message=intent_message)]

    answer_message = MagicMock()
    answer_message.content = "Employees receive 20 vacation days [doc1]."
    answer_message.model_dump.return_value = {
        "role": "assistant",
        "content": answer_message.content,
    }
    answer_choice = MagicMock()
    answer_choice.message = answer_message
    answer_choice.model_dump.return_value = {
        "index": 0,
        "message": answer_message.model_dump.return_value,
    }
    answer_response = MagicMock()
    answer_response.choices = [answer_choice]
    answer_response.model_dump.return_value = {
        "id": "completion-1",
        "choices": [answer_choice.model_dump.return_value],
    }
    openai_client.chat.completions.create.side_effect = [
        intent_response,
        answer_response,
    ]
    client = OYDSearchChatClient(openai_client)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the vacation policy?"}],
        extra_body={"data_sources": [_data_source()]},
    )

    serialized = response.model_dump()
    assert serialized["choices"][0]["message"]["context"]["intent"] == [
        "employee vacation policy"
    ]
    assert serialized["choices"][0]["message"]["context"]["citations"][0]["title"] == "Benefits"


def test_close_delegates_to_openai_client():
    openai_client, _ = _openai_client()
    client = OYDSearchChatClient(openai_client)

    client.close()

    openai_client.close.assert_called_once_with()
