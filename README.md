# Latent-Djezzy

Latent-Djezzy is an experimental multilingual analytics assistant developed as part of an internship contract with **Djezzy**, the Algerian telecommunications company.

The project explores how an AI assistant can help users query business KPI data using natural language, while keeping database access, numerical formatting, and tool execution under deterministic controls.

> This repository represents internship and research/development work. It should not be interpreted as an official Djezzy product, public service, endorsement, or production deployment. Any environment-specific data, credentials, infrastructure details, and internal business information must remain outside the public repository.

## Current implementation

The current implementation is **LatentMind V6**, located under [`v6/`](v6/).

V6 is built as a LangGraph state machine that can:

- understand French, English, Arabic, and Algerian Darija requests
- distinguish greetings, definitions, data questions, unsupported questions, and off-topic requests
- retrieve domain knowledge through RAG
- inspect a database schema and resolve business entities
- generate, validate, and execute SQL
- format KPI values deterministically before natural-language generation
- create chart specifications and report files
- draft emails without sending them automatically
- preserve conversation context across turns
- accept speech input and produce synthesized speech output
- expose the assistant through FastAPI and WebSocket endpoints

For the detailed V6 design, setup instructions, module map, and runtime configuration, see [`v6/README.md`](v6/README.md).

## Why the project was built

Telecommunications analytics often requires users to understand database schemas, metric names, joins, date filters, regional names, and reporting conventions before they can answer a simple business question.

This project investigates a safer assistant workflow:

```text
User question
    |
    v
Intent and action policy
    |
    +--> Domain knowledge retrieval
    +--> Entity and date resolution
    +--> Schema-aware SQL planning
    +--> SQL validation and execution
    +--> Chart, report, or email drafting
    |
    v
Deterministically formatted facts
    |
    v
Natural-language response
```

The goal is not to let a language model freely access a database. The system separates reasoning, validation, execution, and presentation so that important values can be checked before they reach the user.

## Main engineering ideas

### Agentic control loop

V6 replaced a rigid one-pass pipeline with a trained policy loop. The policy selects one action at a time, observes the outcome, and decides whether to continue, retry, switch tools, ask for clarification, or stop.

### Schema-aware SQL

The SQL layer uses live schema information and explicit relationship rules instead of relying only on a prompt that may become outdated. Entity resolution maps user terms such as wilaya names, periods, and business segments to values that actually exist in the database.

### Numerical trust boundary

Business numbers are formatted in Python before the final language-generation step. The model is expected to explain already-prepared values rather than recalculate or reformat raw figures.

This reduces the risk of a small language model changing decimal separators, currencies, percentages, or large KPI values while rewriting the answer.

### Controlled capabilities

Charts, reports, and email drafts are separate tools with explicit inputs and outputs. Email generation creates a draft; sending remains a separate action that requires configuration and intentional execution.

### Multilingual and voice interaction

The project supports text and voice interaction across the languages commonly needed in the internship context. The speech layer normalizes numbers, units, and report references before text-to-speech synthesis.

## Repository layout

```text
Latent-Djezzy/
├── README.md                  # internship context and repository entry point
├── docs/                      # earlier design and research notes
├── v6/                        # current implementation
│   ├── README.md              # complete V6 documentation
│   ├── graph.py               # LangGraph assembly
│   ├── nodes.py               # graph actions
│   ├── brain.py               # trained intent/action/continue policy
│   ├── brain_data.py          # policy training-trace generation
│   ├── train_brain.py         # policy training
│   ├── schema.py              # database schema inspection
│   ├── entities.py            # entity and period resolution
│   ├── sql_tools.py           # SQL generation checks and execution
│   ├── knowledge.py           # retrieval layer
│   ├── numfmt.py              # deterministic number formatting
│   ├── capabilities.py        # chart, report, and email-draft tools
│   ├── speech.py              # speech-to-text and text-to-speech
│   ├── server.py              # FastAPI and WebSocket service
│   ├── benchmark.py           # evaluation harness
│   └── frontend/              # web interface
└── earlier versions/notes     # retained experiments and design history
```

## Project status

This is an evolving internship project and research prototype. The architecture and implementation changed across several iterations as failures were found in routing, schema assumptions, SQL generation, entity resolution, number handling, and voice interaction.

The V6 documentation contains benchmark methodology, but any values explicitly marked as illustrative or placeholder values must not be presented as measured production performance. Only results produced by a repeatable run with recorded configuration should be treated as project metrics.

## Running the current version

Start with the V6 documentation:

```bash
cd v6
```

Then follow [`v6/README.md`](v6/README.md) for:

- dependency installation
- model selection
- policy training
- database configuration
- text and voice benchmarks
- local API serving
- frontend setup

Do not commit database credentials, API tokens, SMTP credentials, ngrok tokens, voice reference files, or internal datasets.

## Data and confidentiality

Because this work originated in a telecommunications internship context:

- public examples should use synthetic, anonymized, or explicitly approved data
- internal database schemas and business definitions should be reviewed before publication
- customer, employee, network, financial, and operational data must not be committed
- credentials and connection information must be supplied through local configuration
- repository documentation should distinguish measured results from examples and placeholders

## Limitations

- The system is a prototype, not a production decision system.
- Text-to-SQL correctness still requires evaluation against the intended question, not only successful SQL execution.
- A trained routing policy can inherit gaps or bias from its generated training traces.
- Voice accuracy depends on language, accent, names, audio quality, and speech-model configuration.
- Local language models and speech models require significant memory and may behave differently across hardware.
- Production deployment would require stronger authentication, authorization, auditing, observability, secret management, and data-governance controls.

## Acknowledgement

This work was carried out as part of an internship contract with **Djezzy in Algeria**. The internship context motivated the focus on multilingual KPI access, telecommunications terminology, database-backed analytics, and practical reporting workflows.
