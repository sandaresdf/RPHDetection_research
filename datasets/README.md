# datasets

All research-generated datasets for this repo, consolidated in one place. External benchmark
clones (PoisonedRAG, EhrAgent, ReAct, agentdriver) and the local Qdrant vector-store cache are
not included here — they stay next to the code that vendors them.

- `prompt_injection/pi_experiments/` — inputs (`benign_tasks.json`, `injection_templates.json`)
  and outputs (`attack_results.json`, `results.jsonl`) for the attack harness in
  `1. primary_rag_agent/app/pi_experiments/`.
- `prompt_injection/batch_runs/` — Anthropic batch request/response files produced by
  `experiement_2/notebooks/prompt_injection_dataset.ipynb`.
- `prompt_injection/experiement_2_prompt_injection_dataset.jsonl` — produced/consumed by
  `experiement_2/app/dataset_generator.py` and `experiement_2/main.py`.
- `prompt_injection/results/` — per-model raw/judged run output from `notebooks/01_prompt_injection.ipynb`.
- `memory_poisoning/strategyqa_raw/`, `memory_poisoning/samples/`, `memory_poisoning/results/` —
  read/written by `notebooks/02_memory_poisoning.ipynb` and `notebooks/05_generate_reasoning_traces.ipynb`.
- `memory_poisoning/generated/` — Anthropic batch outputs from `notebooks/memory_poision_dataset.ipynb`.
- `stealth_attacks/` (incl. `stealth_attacks/results/`) — read/written by
  `notebooks/03_stealth_attacks.ipynb` and `notebooks/generate_stealth_dataset.py`.
- `eval/` — `nq_eval.jsonl` (from `notebooks/memory_poision_dataset.ipynb`) and
  `sample_mcq_results.json`.
- `reasoning_traces/experiement_2/` — written by `experiement_2/app/logger.py`.
- `reasoning_traces/synthetic_traces/` — written by `notebooks/05_generate_reasoning_traces.ipynb`.
