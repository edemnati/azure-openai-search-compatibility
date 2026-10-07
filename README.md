# Azure OpenAI Search Compatibility

OpenAI-compatible Python and TypeScript bridges for applications migrating from Azure
OpenAI On Your Data (OYD). They keep the familiar
`client.chat.completions.create(...)` call, retrieve grounding content directly from Azure
AI Search, and call an Azure OpenAI model deployment such as regional Standard GPT-4o.

> This is a compatibility bridge and migration sample, not a drop-in implementation of
> every internal OYD behavior. Review [compatibility details](docs/compatibility.md) before
> production use.

## Why use it?

- Minimize application-code changes during OYD migration.
- Keep the existing `messages`, `data_sources`, and response-access patterns.
- Preserve `message.context["citations"]` and expose generated search intents.
- Use simple, semantic, vector, and hybrid Azure AI Search queries.
- Compare current OYD and target quality with Microsoft Foundry evaluators.

## Request flow

```text
Conversation
    |
    v
Azure OpenAI intent generation
    |
    v
1-3 Azure AI Search queries
    |
    v
Deduplication + reciprocal-rank fusion
    |
    v
Azure OpenAI grounded answer
    |
    v
OpenAI-style response + OYD-style context
```

## Installation

### Python

The package is currently installed from source:

```powershell
git clone <your-public-repository-url>
cd azure-openai-search-compatibility
python -m pip install -e .
```

For development:

```powershell
python -m pip install -e ".[dev]"
```

### TypeScript

The publishable npm package is under [`typescript/`](typescript/README.md):

```powershell
cd typescript
npm install
npm run build
npm pack
```

See the [TypeScript usage guide](typescript/README.md) for an Azure OpenAI example.

## Minimal migration

### Current OYD call

```python
response = openai_client.chat.completions.create(
    model="<deployment-name>",
    messages=messages,
    extra_body={"data_sources": data_sources},
)
```

### Replacement

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

from azure_openai_search_compat import OYDSearchChatClient

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential,
    "https://cognitiveservices.azure.com/.default",
)
openai_client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com",
    api_version="2024-10-21",
    azure_ad_token_provider=token_provider,
)
client = OYDSearchChatClient(
    openai_client,
    search_credential=credential,
)

response = client.chat.completions.create(
    model="<deployment-name>",
    messages=messages,
    extra_body={"data_sources": data_sources},
)

print(response.choices[0].message.content)
print(response.choices[0].message.context["intent"])
print(response.choices[0].message.context["citations"])
```

The effective change is:

```diff
+ from azure_openai_search_compat import OYDSearchChatClient

  openai_client = AzureOpenAI(...)
+ client = OYDSearchChatClient(openai_client, search_credential=credential)

- response = openai_client.chat.completions.create(...)
+ response = client.chat.completions.create(...)
```

## Data source configuration

Existing OYD-style Azure AI Search configuration is accepted:

```python
data_sources = [
    {
        "type": "azure_search",
        "parameters": {
            "endpoint": "https://<search-service>.search.windows.net",
            "index_name": "<index-name>",
            "authentication": {
                "type": "system_assigned_managed_identity",
            },
            "query_type": "semantic",
            "semantic_configuration": "<semantic-configuration>",
            "fields_mapping": {
                "content_fields": ["content"],
                "title_field": "title",
                "url_field": "url",
                "vector_fields": ["contentVector"],
            },
            "filter": None,
            "strictness": 3,
            "top_n_documents": 5,
            "in_scope": True,
            "role_information": "Answer using the retrieved documents.",
        },
    }
]
```

For vector queries, configure an Azure AI Search vectorizer on the index. The bridge uses
`VectorizableTextQuery`, so the Search service performs text vectorization.

## Authentication and RBAC

Recommended production configuration:

| Identity | Role | Scope |
|---|---|---|
| Application or managed identity | `Cognitive Services OpenAI User` | Azure OpenAI or Foundry resource |
| Application or managed identity | `Search Index Data Reader` | Azure AI Search service |

When the data source authentication type is managed identity, the bridge uses the
`search_credential` passed to `OYDSearchChatClient`. This is normally the application's
identity. OYD instead used the Azure OpenAI resource identity, so RBAC assignments might
need to change during migration.

API-key Search authentication is supported for compatibility but is not recommended for
production.

## Supported behavior

- OpenAI-style synchronous chat-completions call
- Simple and semantic text search
- Vector, vector-simple-hybrid, and vector-semantic-hybrid search
- Conversation-aware search intent generation
- Up to three generated intents with reciprocal-rank result fusion
- Search filters and field mappings
- `role_information` and `in_scope`
- OYD-style citation metadata
- Attribute access and `model_dump()` serialization
- Custom Azure OpenAI deployment names

## Intentional limitations

- Streaming and async calls are not supported.
- Intent generation adds one Azure OpenAI request, latency, and token usage.
- Strictness is approximated with semantic reranker-score filtering. It has no equivalent
  effect when reranker scores are unavailable.
- The bridge does not reproduce undocumented OYD ranking or prompt logic.
- Search results are inserted into the model context and count toward the model token limit.
- Only the `azure_search` data-source type is supported.

See [Compatibility and behavior](docs/compatibility.md) for details.

## Testing

Run local unit tests:

```powershell
python -m pytest -m "not environment and not evaluation"
```

Customer-specific Azure tests belong under [tests/environment](tests/environment/README.md).
Credentials, live golden data, and generated results are ignored by Git.

## Foundry evaluation

The comparison runner executes the current OYD request and the replacement against the same
golden dataset, then scores both with Foundry groundedness, relevance, coherence, fluency,
similarity, and F1 evaluators.

See [Evaluation guide](docs/evaluation.md).

## Project structure

```text
src/azure_openai_search_compat/  Reusable compatibility package
typescript/                      Publishable TypeScript package and tests
tests/unit/                      Offline mocked tests
tests/environment/               Opt-in customer Azure tests
scripts/                         Evaluation tooling
docs/                            Compatibility and evaluation guidance
```

## Security

Do not commit credentials, customer data, endpoints, resource IDs, live golden datasets, or
evaluation outputs. See [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
