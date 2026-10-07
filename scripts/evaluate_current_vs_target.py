"""Compare the current OYD call with the compatibility target using Foundry metrics."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.ai.evaluation import (
    CoherenceEvaluator,
    EvaluationResult,
    F1ScoreEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator,
    evaluate,
)
from azure.core.credentials import AccessToken, TokenCredential
from azure.identity import AzureCliCredential, get_bearer_token_provider
from evaluation_report import write_html_report
from openai import AzureOpenAI

from azure_openai_search_compat import OYDSearchChatClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_TEST_DIR = PROJECT_ROOT / "tests" / "environment"


class StaticTokenCredential:
    """Process-safe credential for parallel evaluator workers."""

    def __init__(self, access_token: AccessToken) -> None:
        self._access_token = access_token

    def get_token(
        self,
        *scopes: str,
        claims: str | None = None,
        tenant_id: str | None = None,
        enable_cae: bool = False,
        **kwargs: Any,
    ) -> AccessToken:
        return self._access_token


def required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the {name} environment variable before running evaluation.")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as dataset_file:
        return [json.loads(line) for line in dataset_file if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=True) + "\n")


def comma_separated_environment(name: str) -> list[str]:
    return [
        value.strip()
        for value in os.environ.get(name, "").split(",")
        if value.strip()
    ]


def search_parameters() -> dict[str, Any]:
    api_key = os.environ.get("AZURE_SEARCH_API_KEY")
    authentication = (
        {"type": "api_key", "key": api_key}
        if api_key
        else {"type": "system_assigned_managed_identity"}
    )
    query_type = os.environ.get("AZURE_SEARCH_QUERY_TYPE", "semantic")
    parameters: dict[str, Any] = {
        "endpoint": required_environment("AZURE_SEARCH_ENDPOINT"),
        "index_name": required_environment("AZURE_SEARCH_INDEX"),
        "authentication": authentication,
        "query_type": query_type,
        "fields_mapping": {
            "content_fields": comma_separated_environment(
                "AZURE_SEARCH_CONTENT_FIELDS"
            ),
            "vector_fields": comma_separated_environment(
                "AZURE_SEARCH_VECTOR_FIELDS"
            ),
        },
        "strictness": int(os.environ.get("AZURE_SEARCH_STRICTNESS", "3")),
        "top_n_documents": int(
            os.environ.get("AZURE_SEARCH_TOP_N_DOCUMENTS", "5")
        ),
        "in_scope": os.environ.get("AZURE_SEARCH_IN_SCOPE", "true").lower()
        == "true",
    }
    if "semantic" in query_type:
        parameters["semantic_configuration"] = required_environment(
            "AZURE_SEARCH_SEMANTIC_CONFIGURATION"
        )
    search_filter = os.environ.get("AZURE_SEARCH_FILTER")
    if search_filter:
        parameters["filter"] = search_filter

    field_mapping = parameters["fields_mapping"]
    for environment_name, mapping_name in (
        ("AZURE_SEARCH_TITLE_FIELD", "title_field"),
        ("AZURE_SEARCH_URL_FIELD", "url_field"),
        ("AZURE_SEARCH_FILEPATH_FIELD", "filepath_field"),
    ):
        value = os.environ.get(environment_name)
        if value:
            field_mapping[mapping_name] = value
    return parameters


def citation_context(citations: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        "\n".join(
            part
            for part in (
                f"Title: {citation.get('title')}" if citation.get("title") else "",
                f"URL: {citation.get('url')}" if citation.get("url") else "",
                f"Content: {citation.get('content', '')}",
            )
            if part
        )
        for citation in citations
    )


def run_current(
    openai_client: AzureOpenAI,
    deployment: str,
    case: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    response = openai_client.chat.completions.create(
        model=deployment,
        messages=case["messages"],
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": parameters,
                }
            ]
        },
        temperature=0,
    )
    message = response.choices[0].message
    context = getattr(message, "context", None) or {}
    citations = context.get("citations", [])
    return {
        "id": case["id"],
        "query": case["query"],
        "ground_truth": case["ground_truth"],
        "response": message.content or "",
        "context": citation_context(citations),
        "intents": context.get("intent", []),
        "citation_count": len(citations),
    }


def run_target(
    target_client: OYDSearchChatClient,
    deployment: str,
    case: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    response = target_client.chat.completions.create(
        model=deployment,
        messages=case["messages"],
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": parameters,
                }
            ]
        },
        temperature=0,
    )
    message = response.choices[0].message
    citations = message.context["citations"]
    return {
        "id": case["id"],
        "query": case["query"],
        "ground_truth": case["ground_truth"],
        "response": message.content or "",
        "context": citation_context(citations),
        "intents": message.context["intent"],
        "citation_count": len(citations),
    }


def build_evaluators(
    model_config: dict[str, Any],
    credential: TokenCredential,
) -> dict[str, Any]:
    return {
        "groundedness": GroundednessEvaluator(model_config, credential=credential),
        "relevance": RelevanceEvaluator(model_config, credential=credential),
        "coherence": CoherenceEvaluator(model_config, credential=credential),
        "fluency": FluencyEvaluator(model_config, credential=credential),
        "similarity": SimilarityEvaluator(model_config, credential=credential),
        "f1_score": F1ScoreEvaluator(),
    }


def evaluate_output(
    *,
    name: str,
    data_path: Path,
    output_path: Path,
    evaluators: dict[str, Any],
    project_endpoint: str,
    credential: AzureCliCredential,
) -> EvaluationResult:
    return evaluate(
        evaluation_name=name,
        data=str(data_path),
        evaluators=evaluators,
        evaluator_config={
            "groundedness": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "context": "${data.context}",
                }
            },
            "relevance": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "coherence": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                }
            },
            "fluency": {
                "column_mapping": {
                    "response": "${data.response}",
                }
            },
            "similarity": {
                "column_mapping": {
                    "query": "${data.query}",
                    "response": "${data.response}",
                    "ground_truth": "${data.ground_truth}",
                }
            },
            "f1_score": {
                "column_mapping": {
                    "response": "${data.response}",
                    "ground_truth": "${data.ground_truth}",
                }
            },
        },
        azure_ai_project=project_endpoint,
        credential=credential,
        output_path=str(output_path),
        fail_on_evaluator_errors=True,
    )


def numeric_metrics(result: EvaluationResult) -> dict[str, float]:
    metrics = result.get("metrics", {})
    return {
        name: float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float))
    }


def compare_metrics(
    current_result: EvaluationResult,
    target_result: EvaluationResult,
) -> dict[str, dict[str, float]]:
    current_metrics = numeric_metrics(current_result)
    target_metrics = numeric_metrics(target_result)
    comparison: dict[str, dict[str, float]] = {}
    for metric in sorted(current_metrics.keys() & target_metrics.keys()):
        comparison[metric] = {
            "current": current_metrics[metric],
            "target": target_metrics[metric],
            "delta": target_metrics[metric] - current_metrics[metric],
        }
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or score current-versus-target evaluation data."
    )
    parser.add_argument(
        "--current-data",
        type=Path,
        help="Precomputed current OYD JSONL data.",
    )
    parser.add_argument(
        "--target-data",
        type=Path,
        help="Precomputed target compatibility JSONL data.",
    )
    parser.add_argument(
        "--timestamp",
        help="Timestamp prefix to use for precomputed evaluation artifacts.",
    )
    args = parser.parse_args()
    precomputed_values = (args.current_data, args.target_data, args.timestamp)
    if any(precomputed_values) and not all(precomputed_values):
        parser.error(
            "--current-data, --target-data, and --timestamp must be provided together"
        )
    return args


def main() -> None:
    args = parse_args()
    credential = AzureCliCredential()
    deployment = required_environment("AZURE_OPENAI_DEPLOYMENT")
    project_endpoint = required_environment("AZURE_AI_PROJECT_ENDPOINT")
    results_dir = Path(
        os.environ.get(
            "EVALUATION_RESULTS_DIR",
            str(ENVIRONMENT_TEST_DIR / "results"),
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.current_data:
        timestamp = args.timestamp
        current_data_path = args.current_data
        target_data_path = args.target_data
        if not current_data_path.is_file() or not target_data_path.is_file():
            raise RuntimeError("Precomputed current and target JSONL files must exist.")
        current_rows = load_jsonl(current_data_path)
        target_rows = load_jsonl(target_data_path)
        current_ids = [row.get("id") for row in current_rows]
        target_ids = [row.get("id") for row in target_rows]
        if current_ids != target_ids:
            raise RuntimeError(
                "Precomputed current and target data must contain matching case IDs."
            )
        cases = current_rows
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        current_data_path = results_dir / f"{timestamp}-current-data.jsonl"
        target_data_path = results_dir / f"{timestamp}-target-data.jsonl"
        dataset_path = Path(
            os.environ.get(
                "EVALUATION_DATASET",
                str(ENVIRONMENT_TEST_DIR / "golden_qa.jsonl"),
            )
        )
        if not dataset_path.is_file():
            raise RuntimeError(
                f"Golden dataset not found: {dataset_path}. Copy "
                "tests/environment/golden_qa.example.jsonl to the ignored "
                "tests/environment/golden_qa.jsonl and add verified answers."
            )
        cases = load_jsonl(dataset_path)
        max_cases = int(os.environ.get("EVALUATION_MAX_CASES", "0"))
        if max_cases > 0:
            cases = cases[:max_cases]
        if not cases:
            raise RuntimeError(
                "The golden evaluation dataset contains no cases to evaluate."
            )
        token_provider = get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
        openai_client = AzureOpenAI(
            azure_endpoint=required_environment("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_ad_token_provider=token_provider,
        )
        target_client = OYDSearchChatClient(
            openai_client, search_credential=credential
        )
        parameters = search_parameters()
        current_rows = [
            run_current(openai_client, deployment, case, parameters)
            for case in cases
        ]
        target_rows = [
            run_target(target_client, deployment, case, parameters)
            for case in cases
        ]
        write_jsonl(current_data_path, current_rows)
        write_jsonl(target_data_path, target_rows)

    model_config = {
        "azure_endpoint": required_environment("AZURE_OPENAI_ENDPOINT"),
        "azure_deployment": deployment,
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    }
    evaluator_credential = StaticTokenCredential(
        credential.get_token("https://cognitiveservices.azure.com/.default")
    )
    evaluators = build_evaluators(model_config, evaluator_credential)
    current_result = evaluate_output(
        name=f"oyd-current-{timestamp}",
        data_path=current_data_path,
        output_path=results_dir / f"{timestamp}-current-evaluation.json",
        evaluators=evaluators,
        project_endpoint=project_endpoint,
        credential=credential,
    )
    target_result = evaluate_output(
        name=f"oyd-target-{timestamp}",
        data_path=target_data_path,
        output_path=results_dir / f"{timestamp}-target-evaluation.json",
        evaluators=evaluators,
        project_endpoint=project_endpoint,
        credential=credential,
    )

    comparison = compare_metrics(current_result, target_result)
    comparison_path = results_dir / f"{timestamp}-comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path = results_dir / f"{timestamp}-report.html"
    write_html_report(
        output_path=report_path,
        timestamp=timestamp,
        current_result=current_result,
        target_result=target_result,
        comparison=comparison,
    )

    print(f"Evaluated {len(cases)} golden cases.")
    print(f"{'Metric':45} {'Current':>10} {'Target':>10} {'Delta':>10}")
    for metric, values in comparison.items():
        print(
            f"{metric:45} "
            f"{values['current']:10.3f} "
            f"{values['target']:10.3f} "
            f"{values['delta']:+10.3f}"
        )
    print(f"Comparison saved to: {comparison_path}")
    print(f"HTML report saved to: {report_path}")
    current_studio_url = current_result.get("studio_url")
    target_studio_url = target_result.get("studio_url")
    if current_studio_url:
        print(f"Current Foundry evaluation: {current_studio_url}")
    if target_studio_url:
        print(f"Target Foundry evaluation: {target_studio_url}")


if __name__ == "__main__":
    main()
