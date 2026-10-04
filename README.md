# Multi-Agent Research Assistant

A four-role research workflow for web research, summarization, evidence checking, and report writing. The roles share a locally loaded Flan-T5 model and exchange intermediate outputs through structured Python state. DuckDuckGo supplies online search results.

## Roles

| Role | Task |
|---|---|
| Researcher | Generate search queries, collect search-result snippets, and synthesize notes |
| Summarizer | Organize research notes into a thematic summary |
| Fact-checker | Extract claims, retrieve evidence snippets, and generate claim verdicts |
| Writer | Produce a report from the summary and verdicts |

## Workflow

1. Accept a research topic.
2. Run the configured research passes; default: two.
3. Summarize accumulated notes.
4. Check up to three extracted claims.
5. Generate and save a Markdown report.

The workflow executes sequentially. Shared state records the topic, notes, summary, evidence-checking results, report, status, and iteration count.

## Components

| Component | Implementation |
|---|---|
| Language model | `google/flan-t5-large` |
| Model execution | Local CPU or CUDA inference |
| Search | DuckDuckGo |
| Role coordination | Python functions and shared state |
| Verdict categories | Supported, disputed, or unverified |
| Output | Timestamped Markdown report |

Local model inference does not require a paid API key. Web research requires internet access.

## Quick Start

```bash
python -m pip install -r requirements.txt
python main.py "Applications of AI in industrial maintenance"
```

Specify an output file:

```bash
python main.py "Retrieval-augmented generation" --output reports/rag_report.md
```

Model and workflow settings are configured in `configs/config.py`.

## Project Structure

| File | Purpose |
|---|---|
| `src/graph.py` | Active role implementations and workflow sequencing |
| `src/tools.py` | Search and evidence-checking helper |
| `main.py` | CLI and report export |
| `configs/config.py` | Model, research-round, and claim-count settings |
| `reports/` | Generated Markdown reports |

## Technologies

Python, PyTorch, Transformers, Flan-T5, and DuckDuckGo Search.
