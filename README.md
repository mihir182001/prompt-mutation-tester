# Prompt Mutation Testing Framework

A developer tool for testing and improving LLM prompt reliability through adversarial mutation testing.

## Overview

Prompts break silently. A model update, a slight rephrasing, or an edge case input can cause your LLM feature to produce wrong outputs with no warning. There is no standard way to test prompts the way we test code.

This framework applies mutation testing concepts from software engineering to LLM prompts. It automatically generates adversarial variants of a prompt, runs them through a language model, evaluates each output against expected behaviour using an LLM judge, and produces a robustness score with failure analysis and auto-generated fixes.

## What Is Built

- **Single prompt testing** — test any prompt against 10 adversarial mutation strategies
- **LLM judge** — objective output scoring using a second AI call
- **Auto fix suggester** — analyses failures and generates an improved prompt automatically
- **Iterative optimizer** — repeatedly fixes and retests until the prompt reaches a target score
- **Autonomous agent** — an LLM agent that adaptively selects which mutations to try next based on previous failures
- **RAG tester** — upload a PDF or text document and test extraction prompt robustness
- **Multi-model comparison** — test the same prompt across Llama 3.3 70B, Llama 3.1 8B and Gemma 2 9B side by side
- **LLM call tracer** — logs every API call with latency, token count and estimated cost
- **Dashboard** — visual analytics including score distribution, trend over time and weakest strategies
- **Batch testing** — test multiple prompts at once, all saved to history
- **History tracking** — all test runs saved to database with timestamps
- **PDF and CSV export** — downloadable reports for stakeholder sharing

## Tech Stack

- Python, Flask, SQLAlchemy, SQLite
- Groq API with Llama 3.3 70B
- Chart.js, Vanilla JavaScript
- ReportLab for PDF generation

## Installation

Clone the repository:

```bash
git clone https://github.com/mihir182001/prompt-mutation-tester
cd prompt-mutation-tester
```

Install dependencies:

```bash
pip install flask flask-sqlalchemy groq rich pypdf reportlab
```

Set your Groq API key — get one free at console.groq.com:

```bash
# Windows
$env:GROQ_API_KEY="your_key_here"

# Mac/Linux
export GROQ_API_KEY=your_key_here
```

Run the app:

```bash
python app.py
```

Open http://127.0.0.1:5000

## Project Structure

```
prompt-mutation-tester/
├── app.py
├── database.py
├── prompt_mutator/
│   ├── mutator.py
│   ├── runner.py
│   ├── judge.py
│   ├── reporter.py
│   ├── fixer.py
│   ├── optimizer.py
│   ├── agent.py
│   ├── rag_tester.py
│   ├── llm_tracer.py
│   └── exporter.py
└── templates/
    └── index.html
```

## Example

```
Prompt: "Summarise this text in exactly 3 bullet points."
Initial score: 43.6/100  FRAGILE

Failures found:
  tone_casual          format_failure   Model added an introductory sentence
  ambiguity_injection  format_failure   Output switched to numbered list
  under_specification  content_failure  Returned variable number of bullets

Auto-generated fix:
"Summarise the text in exactly 3 bullet points with no introductory or
concluding sentences, using bullet points only."

Score after fix: 100/100  ROBUST
```

## Roadmap

- Python SDK — pip installable package for use in any Python project
- GitHub Actions integration — run mutation tests automatically on every commit
- Slack alerts — notify teams when a prompt score drops below a threshold
- Custom mutation strategy plugins — let teams write their own strategies
- On-premise deployment — for enterprise teams with data privacy requirements

## License

MIT