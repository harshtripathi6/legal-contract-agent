# Legal Contract Review Agent

A small research/demo application showing how human-reviewed preferences can
persist through Mubit and improve a later review of a synthetic SaaS contract.
It reviews only limitation-of-liability and indemnification clauses.

This project is not legal advice and does not determine whether contract terms
are legal or illegal. A qualified human reviewer must verify every finding
before anyone acts on it. Use synthetic contracts only.

## Setup

Python 3.11 or newer is required. The project pins the stable public
`mubit-sdk==0.13.2` release and installs from ordinary PyPI.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

Fill in `.env`, then export it in the shell; the application deliberately does
not add a dotenv dependency:

```bash
set -a
source .env
set +a
```

`NVIDIA_API_KEY` is always required. `LLM_MODEL` defaults to
`nvidia/nemotron-3.5-lightning-30b-a3b`, and `LLM_BASE_URL` defaults to NVIDIA's
hosted NIM endpoint. Mubit variables are required only for Memory ON. Use a
fresh synthetic `MUBIT_USER` for each clean demonstration, while keeping it
unchanged across Contract A and B.

The single inference path uses the official OpenAI Python client with NVIDIA's
OpenAI-compatible Chat Completions endpoint. It requests JSON mode, disables
model reasoning for the structured response, and validates the returned JSON
against the strict Pydantic review schema before displaying any finding.

## Demo

First review Contract A with memory enabled:

```bash
contract-review review contracts/contract_a.md --memory on
```

For its limitation-of-liability finding, choose `edited` and enter a clear
client preference such as:

```text
Use a mutual 1x-fees cap; accept 2x as fallback; do not accept uncapped consequential damages.
```

The CLI stores only a compact human-feedback lesson, records the human outcome,
and asks Mubit to reflect. It does not store the contract or raw model prompt.

Then review Contract B twice with the same model and base instructions:

```bash
contract-review compare contracts/contract_b.md
```

The command runs Memory OFF first and Memory ON second, collects verdicts for
both, and prints JSON containing retrieved-memory counts, accepted/edited/
rejected counts, corrections required, and whether fewer corrections were
needed with memory.

To exercise the baseline without any Mubit credentials or calls:

```bash
contract-review review contracts/contract_b.md --memory off
```

## Failure behavior

- Missing, empty, non-UTF-8, NUL-containing, or oversized contracts fail before
  any API call.
- Invalid structured model output or a non-verbatim quote fails before human
  feedback or memory writes.
- Memory ON fails visibly when Mubit retrieval, ingest, outcome recording, or
  reflection fails; it never silently behaves like Memory OFF.
- Output includes run ID, memory mode, retrieved-memory count, findings,
  verdicts, and Mubit operation status without dumping prompts or secrets.

## Tests

The normal suite is deterministic and requires no API keys or network access:

```bash
python -m unittest discover -s tests -v
```

The comparison is a small human-scored demonstration, not a statistical legal
benchmark. Model output can vary even with temperature set to zero.
