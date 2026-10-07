import assert from "node:assert/strict";
import { describe, it } from "node:test";
import type { AzureOpenAI } from "openai";

import {
  ConfigurationError,
  OYDSearchChatClient,
  type ChatMessage,
  type OpenAICompatibleClient,
  type OYDDataSource,
  UnsupportedConfigurationError,
} from "../src/index.js";

const acceptsAzureOpenAI = (client: AzureOpenAI) =>
  new OYDSearchChatClient(client);
void acceptsAzureOpenAI;

function dataSource(overrides: Record<string, unknown> = {}): OYDDataSource {
  return {
    type: "azure_search",
    parameters: {
      endpoint: "https://example.search.windows.net",
      index_name: "customer-index",
      authentication: { type: "api_key", key: "search-key" },
      query_type: "semantic",
      semantic_configuration: "default",
      fields_mapping: {
        content_fields: ["content"],
        title_field: "title",
        url_field: "url",
        vector_fields: ["contentVector"],
      },
      strictness: 3,
      top_n_documents: 2,
      in_scope: true,
      ...overrides,
    },
  };
}

function openAIClient(intents?: string[]) {
  const answer = {
    id: "completion-1",
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: "Grounded answer" },
        finish_reason: "stop",
      },
    ],
  };
  const calls: Record<string, unknown>[] = [];
  let callIndex = 0;
  const create = async (params: Record<string, unknown>) => {
    calls.push(params);
    if (intents && callIndex++ === 0) {
      return {
        choices: [
          { message: { content: JSON.stringify({ intents }) } },
        ],
      };
    }
    return answer;
  };
  let closed = false;
  const client = {
    chat: { completions: { create } },
    close: () => {
      closed = true;
    },
  } as unknown as OpenAICompatibleClient;
  return {
    client,
    calls,
    answer,
    isClosed: () => closed,
  };
}

function searchResults(documents: Array<Record<string, unknown>>) {
  return {
    results: (async function* () {
      for (const document of documents) {
        const { rerankerScore, ...fields } = document;
        yield { document: fields, score: 1, rerankerScore };
      }
    })(),
  };
}

describe("OYDSearchChatClient", () => {
  it("preserves the call shape and adds citations", async () => {
    const searchCalls: unknown[][] = [];
    const search = async (...args: unknown[]) => {
      searchCalls.push(args);
      return searchResults([
        {
          content: "The customer policy allows 20 vacation days.",
          title: "Benefits",
          url: "https://example.test/benefits",
          rerankerScore: 3.5,
        },
      ]);
    };
    const { client: openAI, calls } = openAIClient([
      "employee vacation policy",
    ]);
    const client = new OYDSearchChatClient(openAI, {
      searchClientFactory: () => ({ search }) as never,
    });

    const result = await client.chat.completions.create({
      model: "gpt-4o-canada",
      messages: [{ role: "user", content: "What is the vacation policy?" }],
      extra_body: { data_sources: [dataSource()] },
      temperature: 0.2,
    });

    assert.equal(result.id, "completion-1");
    assert.equal(
      result.choices[0]?.message?.context?.citations[0]?.title,
      "Benefits",
    );
    assert.deepEqual(result.choices[0]?.message?.context?.intent, [
      "employee vacation policy",
    ]);
    assert.equal(searchCalls[0]?.[0], "employee vacation policy");
    assert.deepEqual(searchCalls[0]?.[1], {
      top: 4,
      select: ["content", "title", "url"],
      queryType: "semantic",
      semanticSearchOptions: { configurationName: "default" },
    });
    assert.equal(calls.length, 2);
    assert.deepEqual(calls[0]?.response_format, { type: "json_object" });
    assert.equal(calls[0]?.temperature, 0);
    assert.equal(calls[1]?.temperature, 0.2);
    assert.equal(calls[1]?.data_sources, undefined);
    assert.equal(calls[1]?.extra_body, undefined);
    const answerMessages = calls[1]?.messages as Array<{ content: string }>;
    assert.match(answerMessages[0]?.content ?? "", /20 vacation days/);
  });

  it("delegates unchanged when data_sources are absent", async () => {
    const { client: openAI, calls, answer } = openAIClient();
    const client = new OYDSearchChatClient(openAI);
    const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];

    const result = await client.chat.completions.create({
      model: "gpt-4o",
      messages,
    });

    assert.equal(result, answer);
    assert.deepEqual(calls, [{ model: "gpt-4o", messages }]);
  });

  it("uses Search vectorization for vector hybrid queries", async () => {
    const searchCalls: unknown[][] = [];
    const search = async (...args: unknown[]) => {
      searchCalls.push(args);
      return searchResults([]);
    };
    const { client: openAI } = openAIClient(["company policy"]);
    const client = new OYDSearchChatClient(openAI, {
      searchClientFactory: () => ({ search }) as never,
    });

    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Find the policy" }],
      extra_body: {
        data_sources: [
          dataSource({
            query_type: "vector_simple_hybrid",
            semantic_configuration: null,
          }),
        ],
      },
    });

    assert.equal(searchCalls[0]?.[0], "company policy");
    assert.deepEqual(
      (searchCalls[0]?.[1] as Record<string, unknown>).vectorSearchOptions,
      {
        queries: [
          {
            kind: "text",
            text: "company policy",
            kNearestNeighborsCount: 2,
            fields: ["contentVector"],
          },
        ],
      },
    );
  });

  it("searches multiple intents and fuses duplicate results", async () => {
    const duplicate = {
      content: "Twenty days of annual leave.",
      title: "Leave policy",
      url: "https://example.test/leave",
      rerankerScore: 4,
    };
    let searchCount = 0;
    const { client: openAI } = openAIClient([
      "annual leave allowance",
      "vacation days per year",
    ]);
    const client = new OYDSearchChatClient(openAI, {
      searchClientFactory: () =>
        ({
          search: async () => {
            searchCount += 1;
            return searchResults([duplicate]);
          },
        }) as never,
    });

    const result = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "user", content: "Tell me about employee benefits." },
        { role: "assistant", content: "Which benefit?" },
        { role: "user", content: "How many days do I get?" },
      ],
      extra_body: { data_sources: [dataSource()] },
    });

    assert.equal(searchCount, 2);
    assert.equal(
      result.choices[0]?.message?.context?.citations.length,
      1,
    );
  });

  it("rejects streaming and incomplete semantic configuration", async () => {
    const { client: openAI } = openAIClient(["hello"]);
    const client = new OYDSearchChatClient(openAI);
    await assert.rejects(
      client.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      }),
      UnsupportedConfigurationError,
    );

    await assert.rejects(
      client.chat.completions.create({
        model: "gpt-4o",
        messages: [{ role: "user", content: "Hello" }],
        extra_body: {
          data_sources: [dataSource({ semantic_configuration: null })],
        },
      }),
      ConfigurationError,
    );
  });

  it("delegates close to the wrapped OpenAI client", () => {
    const { client: openAI, isClosed } = openAIClient();
    const client = new OYDSearchChatClient(openAI);
    client.close();
    assert.equal(isClosed(), true);
  });
});
