# TAPIA Claim Checker

Extract → classify → search → compare → annotate, then a human shows where
the agent was wrong.

```text
pipeline/   extract, classify, search, compare, annotate
web/        local UI
verify/     human review and the second-pass challenger
config/     labels, source tiers, prompts
docs/       pipeline notes and the judge pitch
scripts/    OpenRouter client and the Hermes skill installer
skills/     claim-checker skill
data/       essays, uploads, saved runs, and source logs
```

**Open this for judges:**
[`data/runs/2026-09-18-qwen25-3b/annotated.html`](data/runs/2026-09-18-qwen25-3b/annotated.html)

Human catch: [`verify/error_log.md`](verify/error_log.md)  
Pitch: [`docs/judges_summary.md`](docs/judges_summary.md)

## Conflicts we fixed in the new layout

- Brief’s three labels = **classification** (search or not). Badges also include **`misleading`**.
- Reprints of the Shumer essay are **tier 0** and cannot support a claim.
- `verifier_agent.py` only challenges; it does not replace `manual_review.md`.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY='your-key'   # same terminal as the server; do not commit
python3 web/server.py
```

Open http://127.0.0.1:8000

The agent-started server does **not** inherit a key you exported in another tab. Restart the server in the terminal where you ran `export`. You can also put `OPENROUTER_API_KEY=...` in a gitignored `.env` next to this README.

CLI without the webpage (local Ollama). Use the venv so `openai` is installed:

```bash
.venv/bin/python3 pipeline/run.py --essay data/essays/shumer_2026-02-09.md
```

Optional second pass (writes challenges, does not edit verdicts):

```bash
python3 verify/verifier_agent.py \
  --verdicts data/runs/<run_id>/verdicts.json \
  --evidence data/runs/<run_id>/evidence.json \
  --out data/runs/<run_id>/verifier_challenges.json
```
