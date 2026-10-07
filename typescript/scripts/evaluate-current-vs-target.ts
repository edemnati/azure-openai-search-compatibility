import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  AzureCliCredential,
  getBearerTokenProvider,
} from "@azure/identity";
import { AzureOpenAI, type OpenAI } from "openai";

import {
  OYDSearchChatClient,
  type AzureSearchDataSource,
  type ChatMessage,
  type OYDCitation,
} from "../src/index.js";

interface GoldenCase {
  id: string;
  query: string;
  messages: ChatMessage[];
  ground_truth: string;
}

interface EvaluationRow {
  id: string;
  query: string;
  ground_truth: string;
  response: string;
  context: string;
  intents: string[];
  citation_count: number;
}

interface CurrentContext {
  citations?: OYDCitation[];
  intent?: string | string[];
}

const typescriptRoot = process.cwd();
const projectRoot = resolve(typescriptRoot, "..");
const environmentDirectory = resolve(projectRoot, "tests", "environment");
const datasetPath = resolve(
  process.env.EVALUATION_DATASET ??
    resolve(environmentDirectory, "golden_qa.jsonl"),
);
const resultsDirectory = resolve(
  process.env.EVALUATION_RESULTS_DIR ??
    resolve(environmentDirectory, "results"),
);
const maxCases = integerEnvironment("EVALUATION_MAX_CASES", 0, true);
const allCases = loadJsonl<GoldenCase>(datasetPath);
const cases = maxCases > 0 ? allCases.slice(0, maxCases) : allCases;
if (!cases.length) {
  throw new Error("The golden evaluation dataset contains no cases.");
}

const now = new Date().toISOString();
const timestamp = `${now.slice(0, 10).replaceAll("-", "")}-${now
  .slice(11, 19)
  .replaceAll(":", "")}`;
mkdirSync(resultsDirectory, { recursive: true });
const currentDataPath = resolve(
  resultsDirectory,
  `${timestamp}-current-data.jsonl`,
);
const targetDataPath = resolve(
  resultsDirectory,
  `${timestamp}-target-data.jsonl`,
);

const credential = new AzureCliCredential();
const tokenProvider = getBearerTokenProvider(
  credential,
  "https://cognitiveservices.azure.com/.default",
);
const openAIClient = new AzureOpenAI({
  endpoint: requiredEnvironment("AZURE_OPENAI_ENDPOINT"),
  apiVersion: process.env.AZURE_OPENAI_API_VERSION ?? "2024-10-21",
  azureADTokenProvider: tokenProvider,
});
const targetClient = new OYDSearchChatClient(openAIClient, {
  searchCredential: credential,
});
const deployment = requiredEnvironment("AZURE_OPENAI_DEPLOYMENT");
const dataSource = searchDataSource();

try {
  const currentRows: EvaluationRow[] = [];
  const targetRows: EvaluationRow[] = [];
  for (const evaluationCase of cases) {
    currentRows.push(await runCurrent(evaluationCase));
  }
  for (const evaluationCase of cases) {
    targetRows.push(await runTarget(evaluationCase));
  }
  writeJsonl(currentDataPath, currentRows);
  writeJsonl(targetDataPath, targetRows);
} finally {
  targetClient.close();
}

const scorer = spawnSync(
  process.env.PYTHON ?? "python",
  [
    resolve(projectRoot, "scripts", "evaluate_current_vs_target.py"),
    "--current-data",
    currentDataPath,
    "--target-data",
    targetDataPath,
    "--timestamp",
    timestamp,
  ],
  {
    cwd: projectRoot,
    env: process.env,
    stdio: "inherit",
  },
);
if (scorer.error) {
  throw scorer.error;
}
if (scorer.status !== 0) {
  throw new Error(`Foundry scoring failed with exit code ${scorer.status}.`);
}

async function runCurrent(evaluationCase: GoldenCase): Promise<EvaluationRow> {
  const request: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming & {
    data_sources: AzureSearchDataSource[];
  } = {
    model: deployment,
    messages: evaluationCase.messages,
    data_sources: [dataSource],
    temperature: 0,
  };
  const response = await openAIClient.chat.completions.create(request);
  const message = response.choices[0]?.message as unknown as
    | { content?: unknown; context?: CurrentContext }
    | undefined;
  if (!message || typeof message.content !== "string") {
    throw new Error(`Current OYD returned no answer for case '${evaluationCase.id}'.`);
  }
  const context = message.context ?? {};
  const citations = Array.isArray(context.citations) ? context.citations : [];
  return evaluationRow(evaluationCase, message.content, citations, context.intent);
}

async function runTarget(evaluationCase: GoldenCase): Promise<EvaluationRow> {
  const response = await targetClient.chat.completions.create({
    model: deployment,
    messages: evaluationCase.messages,
    extra_body: { data_sources: [dataSource] },
    temperature: 0,
  });
  const message = response.choices[0]?.message;
  if (!message || typeof message.content !== "string") {
    throw new Error(
      `TypeScript target returned no answer for case '${evaluationCase.id}'.`,
    );
  }
  return evaluationRow(
    evaluationCase,
    message.content,
    message.context?.citations ?? [],
    message.context?.intent,
  );
}

function evaluationRow(
  evaluationCase: GoldenCase,
  response: string,
  citations: OYDCitation[],
  intents: string | string[] | undefined,
): EvaluationRow {
  return {
    id: evaluationCase.id,
    query: evaluationCase.query,
    ground_truth: evaluationCase.ground_truth,
    response,
    context: citationContext(citations),
    intents: Array.isArray(intents) ? intents : intents ? [intents] : [],
    citation_count: citations.length,
  };
}

function searchDataSource(): AzureSearchDataSource {
  const queryType = process.env.AZURE_SEARCH_QUERY_TYPE ?? "semantic";
  return {
    type: "azure_search",
    parameters: {
      endpoint: requiredEnvironment("AZURE_SEARCH_ENDPOINT"),
      index_name: requiredEnvironment("AZURE_SEARCH_INDEX"),
      authentication: process.env.AZURE_SEARCH_API_KEY
        ? { type: "api_key", key: process.env.AZURE_SEARCH_API_KEY }
        : { type: "system_assigned_managed_identity" },
      query_type: queryType,
      ...(queryType.includes("semantic")
        ? {
            semantic_configuration: requiredEnvironment(
              "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
            ),
          }
        : {}),
      fields_mapping: {
        content_fields: commaSeparatedEnvironment(
          "AZURE_SEARCH_CONTENT_FIELDS",
        ),
        vector_fields: commaSeparatedEnvironment(
          "AZURE_SEARCH_VECTOR_FIELDS",
        ),
        ...optionalField("title_field", "AZURE_SEARCH_TITLE_FIELD"),
        ...optionalField("url_field", "AZURE_SEARCH_URL_FIELD"),
        ...optionalField("filepath_field", "AZURE_SEARCH_FILEPATH_FIELD"),
      },
      ...optionalField("filter", "AZURE_SEARCH_FILTER"),
      strictness: integerEnvironment("AZURE_SEARCH_STRICTNESS", 3),
      top_n_documents: integerEnvironment(
        "AZURE_SEARCH_TOP_N_DOCUMENTS",
        5,
      ),
      in_scope:
        (process.env.AZURE_SEARCH_IN_SCOPE ?? "true").toLowerCase() === "true",
    },
  };
}

function citationContext(citations: OYDCitation[]): string {
  return citations
    .map((citation) =>
      [
        citation.title ? `Title: ${citation.title}` : "",
        citation.url ? `URL: ${citation.url}` : "",
        `Content: ${citation.content}`,
      ]
        .filter(Boolean)
        .join("\n"),
    )
    .join("\n\n");
}

function loadJsonl<T>(path: string): T[] {
  return readFileSync(path, "utf8")
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as T);
}

function writeJsonl(path: string, rows: EvaluationRow[]): void {
  writeFileSync(
    path,
    `${rows.map((row) => JSON.stringify(row)).join("\n")}\n`,
    "utf8",
  );
}

function requiredEnvironment(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Set the ${name} environment variable.`);
  }
  return value;
}

function commaSeparatedEnvironment(name: string): string[] {
  return (process.env[name] ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

function integerEnvironment(
  name: string,
  fallback: number,
  allowZero = false,
): number {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isInteger(value) || value < 0 || (!allowZero && value === 0)) {
    throw new Error(
      `${name} must be ${allowZero ? "a non-negative" : "a positive"} integer.`,
    );
  }
  return value;
}

function optionalField(
  property: string,
  environmentName: string,
): Record<string, string> {
  const value = process.env[environmentName];
  return value ? { [property]: value } : {};
}
