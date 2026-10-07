# Environment tests

This folder is for tests against your Azure OpenAI, Microsoft Foundry, and Azure AI Search
resources. Unit tests do not require Azure access.

## Authentication

Run `az login` and grant the test identity:

- `Cognitive Services OpenAI User` on the Azure OpenAI or Foundry resource
- `Search Index Data Reader` on the Azure AI Search service

Managed identity and Microsoft Entra ID are recommended. The committed examples do not
contain credentials. If API-key Search authentication is required for a test, set
`AZURE_SEARCH_API_KEY` only in the current process or in the ignored `.env` file.

## Configure

Copy `.env.example` to `.env` and replace placeholders. `.env` and `golden_qa.jsonl` are
ignored by Git.

For PowerShell, values can be set directly:

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "<deployment-name>"
$env:AZURE_SEARCH_ENDPOINT = "https://<search-service>.search.windows.net"
$env:AZURE_SEARCH_INDEX = "<index-name>"
$env:AZURE_SEARCH_SEMANTIC_CONFIGURATION = "<semantic-configuration>"
$env:AZURE_SEARCH_CONTENT_FIELDS = "content"
$env:AZURE_SEARCH_TITLE_FIELD = "title"
$env:TEST_QUERY = "<known-answer-question>"
$env:EXPECTED_TEXT = "<expected-answer-fragment>"
```

## Run

```powershell
python -m pytest tests/environment -m environment
```

Or use the helper that loads `.env` first:

```powershell
.\tests\environment\run.ps1
```

To run the equivalent TypeScript live test:

```powershell
.\typescript\run-live-test.ps1
```

When `AZURE_SEARCH_USE_ADMIN_KEY=true`, the helper obtains the admin key through Azure CLI
and keeps it only in process memory. Prefer managed identity for production.

The test is skipped when required variables are absent. It fails if Search returns no
citations, Azure OpenAI returns no answer, or `EXPECTED_TEXT` is not present.

## Golden evaluation data

Copy `golden_qa.example.jsonl` to the ignored `golden_qa.jsonl` and replace the examples
with questions and verified answers from your own index. Never commit customer data.

Run the current-versus-target Foundry evaluation with:

```powershell
.\tests\environment\run-evaluation.ps1
```
