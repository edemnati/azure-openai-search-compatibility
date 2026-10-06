# Compatibility and behavior

## OYD behavior mapping

| OYD behavior | Compatibility bridge | Notes |
|---|---|---|
| `chat.completions.create` | Preserved | Synchronous calls only |
| `messages` | Preserved | User and assistant history drives intent generation |
| `data_sources` | Consumed by bridge | Not forwarded to Azure OpenAI |
| Intent generation | Reimplemented | GPT-4o produces one to three standalone queries |
| Search retrieval | Explicit SDK call | Uses `azure-search-documents` |
| Multi-intent ranking | Reciprocal-rank fusion | Not the undocumented OYD ranking algorithm |
| `context.citations` | Preserved | Content, title, URL, filepath, and chunk ID |
| `context.intent` | Preserved as a list | Contains generated standalone queries |
| `role_information` | Preserved | Applied only to grounded answer generation |
| `in_scope` | Preserved | Instructs the model to decline unsupported answers |
| `strictness` | Approximation | Filters semantic reranker scores using `strictness - 1` |
| `top_n_documents` | Preserved per source | Applied after multi-intent fusion |
| Search filter | Preserved | Passed as an OData filter |
| Streaming | Unsupported | Fails explicitly |

## Intent generation

Intent generation receives only user and assistant conversation turns. It does not receive
`role_information` or retrieved documents. This mirrors documented OYD behavior and helps
resolve follow-up questions such as:

```text
User: Tell me about the leave policy.
Assistant: What would you like to know?
User: How many days do I receive?
```

The generated intent might be:

```text
employee annual leave days
```

Intent generation is enabled by default. It adds one model call for each request containing
`data_sources`.

## Search authentication change

For managed-identity authentication:

- OYD used the Azure OpenAI resource identity to access Search.
- The bridge uses the credential supplied by the application.

Assign `Search Index Data Reader` to the new principal before cutover. If a user-assigned
managed identity is required, construct the credential with its client ID and pass that
credential as `search_credential`.

## Query types

| Query type | Search text | Vector query | Semantic reranking |
|---|---:|---:|---:|
| `simple` | Yes | No | No |
| `semantic` | Yes | No | Yes |
| `vector` | No | Yes | No |
| `vector_simple_hybrid` | Yes | Yes | No |
| `vector_semantic_hybrid` | Yes | Yes | Yes |

Vector modes require:

- One or more fields in `fields_mapping.vector_fields`
- A vectorizer configured on the Search index
- A Search API version supported by `azure-search-documents>=11.6.0`

## Response compatibility

The returned proxy delegates standard response attributes to the wrapped OpenAI response
and adds OYD-style context:

```python
message = response.choices[0].message
message.content
message.context["intent"]
message.context["citations"]
```

`response.model_dump()` includes the added context. Code that depends on exact OpenAI SDK
class identity, private SDK attributes, streaming objects, or async clients is not
compatible.

## Security considerations

Retrieved documents are presented to the model as untrusted reference data and not as
instructions. This reduces, but does not eliminate, indirect prompt-injection risk.
Production applications should also use:

- Search index access controls and filters
- Least-privilege managed identities
- Content filtering and application-level authorization
- Evaluation cases for prompt injection and data leakage
- Logging that excludes document content and credentials
