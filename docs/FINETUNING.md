# Fine-tuning & Evaluating Qwen3.5-0.8B on dataagent datasets

## Input format

`data/output/{train,validation,test}.jsonl` — one conversation per line:

```json
{
  "messages": [
    {"role": "system", "content": "Generate valid Dremio SQL ..."},
    {"role": "user", "content": "Schema: ...\nQuestion: ..."},
    {"role": "assistant", "content": "SELECT SUM(amount) ... LIMIT 1"}
  ]
}
```

QC metadata lives in the parallel `*.manifest.jsonl` files and is **never**
inside the assistant target.

## 1. Fine-tune with LLaMA-Factory

```bash
pip install llamafactory[torch]
```

`configs/qwen35_sft.yaml` (LLaMA-Factory YAML):

```yaml
model_name_or_path: Qwen/Qwen3.5-0.8B   # exact HF repo id must be verified at run time
stage: sft
template: qwen                          # chat template id may differ per release; check
finetuning_type: full                   # or lora: lora_rank: 8, lora_target: q_proj,k_proj,v_proj,o_proj
dataset_dir: ../data/output
dataset: train                          # LLaMA-Factory dataset name -> file mapping in dataset_info.json
cutoff_len: 4096
learning_rate: 1.0e-4
num_train_epochs: 3
batch_size: 8
gradient_accumulation_steps: 2
warmup_ratio: 0.03
logging_steps: 10
save_steps: 200
output_dir: ../models/qwen35-sft-dremio
```

Register the dataset in `data/output/dataset_info.json`:

```json
{
  "train": {
    "file_name": "train.jsonl",
    "columns": ["messages"],
    "formatting": "sharegpt",
    "messages": "messages"
  },
  "validation": { "file_name": "validation.jsonl", "columns": ["messages"],
                  "formatting": "sharegpt", "messages": "messages" }
}
```

Run:

```bash
llamafactory-cli train configs/qwen35_sft.yaml
```

## 2. Evaluate on the held-out test split

Two complementary evaluations:

### a) Execution accuracy (preferred)

1. For each test example, reconstruct the prompt and generate SQL with the
   fine-tuned model (greedy decoding).
2. Submit each generated SQL through the *same* validation/execution stack
   (`dataagent.validation`) against the read-only test Dremio.
3. Score: query produced a COMPLETED job AND result equals the reference
   result (compare normalized rows from the manifest).

```bash
python scripts/eval_test.py predictions.jsonl data/output/test.manifest.jsonl
```

Report exact-execution accuracy and validation-failure rate per category.

### b) LLM-review scoring (optional)

Use the semantic reviewer prompt from `generation/prompts.py` with the GLM
model to grade generated SQL against question + schema, giving a 0-1 score
correlated with manual correctness checks.

## 3. Iteration guidance

- If execution accuracy is low in one category (e.g. `date_timestamp`),
  generate additional verified examples for that category and retrain.
- Keep the leakage-safe split: never fold `needs_review` or unverified
  examples into training.
- Track per-version manifest fields (`dataset_version`, fingerprints) so any
  regression can be attributed to a dataset change.
