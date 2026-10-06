# Foundry evaluation guide

The evaluation runner compares:

1. The current Azure OpenAI OYD `data_sources` request
2. The compatibility bridge using explicit Azure AI Search retrieval

Both paths use the same model deployment, messages, Search index, field mappings, query
type, filters, strictness, and document count.

## Metrics

The runner uses Microsoft Foundry evaluators:

- Groundedness
- Relevance
- Coherence
- Fluency
- Similarity to the golden answer
- F1

The two evaluation runs are uploaded to a Foundry project and local row-level artifacts are
written to the ignored `tests/environment/results/` folder.

## Install

```powershell
python -m pip install -e ".[evaluation]"
az login
```

## Create a golden dataset

Copy the committed template:

```powershell
Copy-Item tests\environment\golden_qa.example.jsonl `
  tests\environment\golden_qa.jsonl
```

Replace every question and answer with verified content from your Search index. Include:

- Direct factual questions
- Paraphrases
- Multi-turn follow-ups
- Questions with no answer in the index
- Business-critical edge cases

`golden_qa.jsonl` is ignored by Git because it may contain customer data.

## Configure

Set the variables documented in
[tests/environment/.env.example](../tests/environment/.env.example), including:

```powershell
$env:AZURE_AI_PROJECT_ENDPOINT = "https://<resource>.services.ai.azure.com/api/projects/<project>"
$env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "<deployment-name>"
$env:AZURE_SEARCH_ENDPOINT = "https://<service>.search.windows.net"
$env:AZURE_SEARCH_INDEX = "<index>"
$env:AZURE_SEARCH_CONTENT_FIELDS = "content"
$env:AZURE_SEARCH_TITLE_FIELD = "title"
$env:AZURE_SEARCH_URL_FIELD = "url"
$env:AZURE_SEARCH_QUERY_TYPE = "semantic"
$env:AZURE_SEARCH_SEMANTIC_CONFIGURATION = "<configuration>"
```

If `AZURE_SEARCH_API_KEY` is absent, both paths request managed-identity Search
authentication. Ensure the Azure OpenAI resource identity can access Search for the current
OYD path and the caller identity can access Search for the target path.

## Smoke test

```powershell
$env:EVALUATION_MAX_CASES = "1"
python scripts\evaluate_current_vs_target.py
```

## Full evaluation

```powershell
Remove-Item Env:EVALUATION_MAX_CASES -ErrorAction SilentlyContinue
python scripts\evaluate_current_vs_target.py
```

Optional overrides:

| Variable | Purpose |
|---|---|
| `EVALUATION_DATASET` | Alternate golden JSONL path |
| `EVALUATION_RESULTS_DIR` | Alternate local output directory |
| `EVALUATION_MAX_CASES` | Limit evaluated rows |

## Interpreting results

Use row-level results as well as aggregate scores. In particular, groundedness judges can
score a correct absence statement poorly because the missing fact is not explicitly
present in context. Pair judge metrics with golden-answer similarity and manual review for
unsupported-answer cases.

Do not approve a migration based only on aggregate averages. Define pass thresholds for
business-critical rows and investigate every regression.
