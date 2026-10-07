# Foundry evaluation guide

The evaluation harness is written in Python and remains the reference migration-quality
gate for both package implementations. The Python and TypeScript packages use the same
retrieval, intent-generation, reciprocal-rank fusion, grounding, and citation behavior.
Run the TypeScript package's unit and build checks separately before evaluating quality:

```powershell
Set-Location typescript
npm install
npm run check
npm test
npm run build
Set-Location ..
```

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

Each completed run also writes a self-contained `*-report.html` file. The report includes
aggregate current-versus-target metrics and per-case scores identified by case ID. It
intentionally excludes questions, answers, ground truth, retrieved context, intents,
endpoints, and credentials.

The runner obtains one short-lived Cognitive Services token before starting parallel
evaluator workers. This avoids concurrent Azure CLI subprocess authentication and does not
persist the token.

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

To evaluate the actual TypeScript target implementation, run:

```powershell
.\typescript\run-evaluation.ps1
```

This generates both the current OYD and target response data in TypeScript, then invokes
the same Python Foundry scoring and privacy-conscious HTML reporting stage.

Optional overrides:

| Variable | Purpose |
|---|---|
| `EVALUATION_DATASET` | Alternate golden JSONL path |
| `EVALUATION_RESULTS_DIR` | Alternate local output directory |
| `EVALUATION_MAX_CASES` | Limit evaluated rows |

## HTML report

Open the generated report from `tests/environment/results/` in a browser. It contains:

- The number of golden cases
- Counts of improved, tied, and regressed aggregate metrics
- Current, target, delta, and pass rate for each aggregate metric
- Current, target, and delta scores for every case ID

The report is generated automatically only after both Foundry evaluations complete
successfully. Evaluation artifacts and reports remain ignored by Git because case IDs may
still be customer-specific.

## Interpreting results

Use row-level results as well as aggregate scores. In particular, groundedness judges can
score a correct absence statement poorly because the missing fact is not explicitly
present in context. Pair judge metrics with golden-answer similarity and manual review for
unsupported-answer cases.

Do not approve a migration based only on aggregate averages. Define pass thresholds for
business-critical rows and investigate every regression.
