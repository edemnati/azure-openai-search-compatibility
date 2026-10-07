# Azure OpenAI Search Compatibility for TypeScript

A TypeScript bridge for applications migrating from Azure OpenAI On Your Data (OYD).
It preserves the native asynchronous `client.chat.completions.create(...)` shape,
performs explicit Azure AI Search retrieval, and adds OYD-style citation context to the
OpenAI response.

## Install

The package currently ships from this repository as a tarball. Node.js 20 or newer and
OpenAI 6.x are required:

```powershell
Set-Location typescript
npm install
npm run build
npm pack
```

Install the generated tarball in the customer application:

```powershell
npm install C:\path\to\azure-openai-search-compatibility-0.1.0.tgz
```

After the package is published to npm, installation becomes:

```powershell
npm install azure-openai-search-compatibility openai
npm install --save-dev @types/node
```

## Prerequisites

- An Azure OpenAI deployment supported by the OpenAI JavaScript SDK
- Node.js type declarations (`@types/node`) for strict TypeScript compilation
- An Azure AI Search index containing the configured content fields
- A semantic configuration for semantic query modes
- A Search vectorizer and vector fields for vector query modes
- `Cognitive Services OpenAI User` on the Azure OpenAI resource
- `Search Index Data Reader` on the Azure AI Search service

Use managed identity or another Microsoft Entra credential in production. Search API keys
are supported for migration compatibility but are not recommended.

## Usage

```typescript
import { DefaultAzureCredential, getBearerTokenProvider } from "@azure/identity";
import { AzureOpenAI } from "openai";
import { OYDSearchChatClient, type OYDDataSource } from "azure-openai-search-compatibility";

const credential = new DefaultAzureCredential();
const azureADTokenProvider = getBearerTokenProvider(
  credential,
  "https://cognitiveservices.azure.com/.default",
);
const openAIClient = new AzureOpenAI({
  endpoint: "https://<resource>.openai.azure.com",
  apiVersion: "2024-10-21",
  azureADTokenProvider,
});
const client = new OYDSearchChatClient(openAIClient, {
  searchCredential: credential,
});

const dataSources: OYDDataSource[] = [
  {
    type: "azure_search",
    parameters: {
      endpoint: "https://<search-service>.search.windows.net",
      index_name: "<index-name>",
      authentication: {
        type: "system_assigned_managed_identity",
      },
      query_type: "semantic",
      semantic_configuration: "<semantic-configuration>",
      fields_mapping: {
        content_fields: ["content"],
        title_field: "title",
        url_field: "url",
        vector_fields: ["contentVector"],
      },
      strictness: 3,
      top_n_documents: 5,
      in_scope: true,
    },
  },
];

const response = await client.chat.completions.create({
  model: "<deployment-name>",
  messages: [{ role: "user", content: "What is the vacation policy?" }],
  extra_body: { data_sources: dataSources },
});

console.log(response.choices[0]?.message.content);
console.log(response.choices[0]?.message.context?.intent);
console.log(response.choices[0]?.message.context?.citations);
```

The wrapper also accepts `data_sources` directly on the create parameters. When no data
sources are present, it delegates the request to the wrapped OpenAI client.

## Minimal migration

```diff
  const openAIClient = new AzureOpenAI(...);
+ const client = new OYDSearchChatClient(openAIClient, {
+   searchCredential: credential,
+ });

- const response = await openAIClient.chat.completions.create({
+ const response = await client.chat.completions.create({
    model,
    messages,
-   data_sources: dataSources,
+   extra_body: { data_sources: dataSources },
  });
```

The wrapper also accepts `data_sources: dataSources` directly. `extra_body` is supported
to ease migration from Python applications that share OYD request configuration.

## Configuration reference

| Option | Required | Description |
|---|---:|---|
| `endpoint` | Yes | Azure AI Search endpoint |
| `index_name` | Yes | Search index name |
| `authentication` | Yes | Managed identity type or `api_key` with `key` |
| `fields_mapping.content_fields` | Yes | Fields combined into retrieved document content |
| `query_type` | No | `simple`, `semantic`, `vector`, `vector_simple_hybrid`, or `vector_semantic_hybrid`; defaults to `simple` |
| `semantic_configuration` | Semantic modes | Search semantic configuration name |
| `fields_mapping.vector_fields` | Vector modes | Fields queried through the index vectorizer |
| `filter` | No | OData filter passed to Search |
| `strictness` | No | Approximate semantic reranker threshold; defaults to `3` |
| `top_n_documents` | No | Maximum fused citations per source; defaults to `5` |
| `in_scope` | No | Instructs the model to answer only from retrieved documents |
| `role_information` | No | Additional grounded-answer instructions |

`OYDSearchChatClient` also accepts:

| Client option | Default | Description |
|---|---:|---|
| `searchCredential` | `DefaultAzureCredential` | Credential used for managed-identity Search access |
| `maxContextCharacters` | `32000` | Maximum retrieved content characters sent to the model |

## Response and errors

The returned value preserves the OpenAI `ChatCompletion` fields. Grounded responses add:

```typescript
response.choices[0]?.message.context?.intent;
response.choices[0]?.message.context?.citations;
```

The package throws:

- `ConfigurationError` for missing or invalid Search configuration and malformed intent
  responses
- `UnsupportedConfigurationError` for streaming, unsupported data-source types,
  authentication modes, or query modes

Calls are Promise-based. Streaming is intentionally unsupported and fails explicitly.

## Supported behavior and limitations

Supported behavior includes conversation-aware generation of up to three Search intents,
simple/semantic/vector/hybrid retrieval, reciprocal-rank fusion, filters, field mappings,
role information, in-scope grounding, and OYD-style citations.

The bridge does not reproduce undocumented OYD ranking or prompt logic. Intent generation
adds one model request, retrieved content counts toward the model context, and strictness
only filters results when semantic reranker scores are available. See the repository
[compatibility guide](../docs/compatibility.md) for details.

## Development

```powershell
npm install
npm run check
npm test
npm run build
npm pack --dry-run
```

The package is ESM-only.

## Live Azure test

Configure the repository's ignored `tests/environment/.env` file, sign in with
`az login`, and run:

```powershell
.\run-live-test.ps1
```

The test uses `AzureCliCredential` for Azure OpenAI and, unless a Search API key is
configured, Azure AI Search. It requires `Cognitive Services OpenAI User` and
`Search Index Data Reader` for the signed-in identity. The test checks that the response
contains an answer, generated intent, and at least one citation without printing customer
content.

## Current-versus-target evaluation

To generate both current OYD and target rows with TypeScript, score them with the Foundry
evaluators, and produce the JSON and HTML comparison reports:

```powershell
.\run-evaluation.ps1
```

The TypeScript process performs all Azure OpenAI and Search calls for both solutions. It
then invokes the repository's Python Foundry scoring stage because the evaluator SDK is
Python-based.

## Production checklist

Before customer production use:

1. Run the live test against the customer's Search index.
2. Replace the example golden data with verified customer cases.
3. Run the current-versus-target evaluation and review per-case regressions.
4. Validate authorization filters and prompt-injection cases.
5. Confirm latency, token usage, and context-size limits for the workload.
6. Pin an approved package version and deploy with least-privilege identities.
