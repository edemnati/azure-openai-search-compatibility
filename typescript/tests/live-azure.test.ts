import assert from "node:assert/strict";
import { it } from "node:test";

import {
  AzureCliCredential,
  getBearerTokenProvider,
} from "@azure/identity";
import { AzureOpenAI } from "openai";

import {
  OYDSearchChatClient,
  type AzureSearchDataSource,
} from "../src/index.js";

const requiredVariables = [
  "AZURE_OPENAI_ENDPOINT",
  "AZURE_OPENAI_DEPLOYMENT",
  "AZURE_SEARCH_ENDPOINT",
  "AZURE_SEARCH_INDEX",
  "AZURE_SEARCH_CONTENT_FIELDS",
  "TEST_QUERY",
] as const;
const missingVariables = requiredVariables.filter(
  (name) => !process.env[name],
);

it(
  "returns a grounded answer from live Azure resources",
  {
    skip: missingVariables.length
      ? `Missing environment variables: ${missingVariables.join(", ")}`
      : false,
  },
  async () => {
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
    const client = new OYDSearchChatClient(openAIClient, {
      searchCredential: credential,
    });
    const dataSource: AzureSearchDataSource = {
      type: "azure_search",
      parameters: {
        endpoint: requiredEnvironment("AZURE_SEARCH_ENDPOINT"),
        index_name: requiredEnvironment("AZURE_SEARCH_INDEX"),
        authentication: process.env.AZURE_SEARCH_API_KEY
          ? { type: "api_key", key: process.env.AZURE_SEARCH_API_KEY }
          : { type: "system_assigned_managed_identity" },
        query_type: process.env.AZURE_SEARCH_QUERY_TYPE ?? "semantic",
          ...optionalField(
            "semantic_configuration",
            "AZURE_SEARCH_SEMANTIC_CONFIGURATION",
          ),
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
          (process.env.AZURE_SEARCH_IN_SCOPE ?? "true").toLowerCase() ===
          "true",
      },
    };

    try {
      const response = await client.chat.completions.create({
        model: requiredEnvironment("AZURE_OPENAI_DEPLOYMENT"),
        messages: [
          { role: "user", content: requiredEnvironment("TEST_QUERY") },
        ],
        extra_body: { data_sources: [dataSource] },
        temperature: 0,
      });
      const message = response.choices[0]?.message;
      if (!message || typeof message.content !== "string") {
        assert.fail("Azure OpenAI returned no text response.");
      }
      const content = message.content;
      assert.ok(content);
      assert.ok(message.context?.intent.length);
      assert.ok(message.context?.citations.length);

      const expectedText = process.env.EXPECTED_TEXT;
      if (expectedText) {
        assert.match(
          content.toLocaleLowerCase(),
          new RegExp(escapeRegExp(expectedText.toLocaleLowerCase())),
        );
      }
    } finally {
      client.close();
    }
  },
);

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

function integerEnvironment(name: string, fallback: number): number {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer.`);
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

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
