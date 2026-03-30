# GLiNER Contamination-Free Retraining Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt hf-autoresearch for GLiNER fine-tuning on a deduped v6-clean train split, with vanilla Python GLiNER verification against the frozen holdout.

**Architecture:** Three scripts in the repo root: `prepare_data.py` (dedup v6-clean train against frozen holdout), `train.py` (GLiNER fine-tuning via HF Jobs), `program.md` (agent loop instructions). The agent loop metric is nervaluate strict F1 on `arthrod/pii-gliner-evals` (49,952 rows), computed by running vanilla `GLiNER.from_pretrained()` inference.

**Tech Stack:** Python 3.11+, gliner, torch, nervaluate, datasets, huggingface_hub, datasketch (MinHash fuzzy dedup)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `prepare_data.py` | Downloads v6-clean train, loads frozen holdout, runs exact+fuzzy dedup, outputs clean parquet to HF bucket |
| `train.py` | Self-contained GLiNER fine-tuning script for HF Jobs. Loads clean data from mounted bucket, trains, saves checkpoint |
| `evaluate.py` | Loads a checkpoint with vanilla GLiNER, runs inference on frozen holdout, computes nervaluate strict F1 |
| `program.md` | Agent instructions: research → train → evaluate → keep/discard loop |
| `tests/test_prepare_data.py` | Tests for dedup logic |
| `tests/test_evaluate.py` | Tests for evaluation pipeline |

---

### Task 1: Data Preparation Script — Exact Dedup

**Files:**
- Create: `prepare_data.py`
- Create: `tests/test_prepare_data.py`

- [ ] **Step 1: Write failing test for exact dedup**

```python
# tests/test_prepare_data.py
import pytest

def test_exact_dedup_removes_matching_sample_ids():
    """Train rows with sample_ids present in holdout must be removed."""
    from prepare_data import exact_dedup

    train_rows = [
        {"sample_id": "aaa", "tokenized_text": ["hello"], "ner": [], "source": "test"},
        {"sample_id": "bbb", "tokenized_text": ["world"], "ner": [], "source": "test"},
        {"sample_id": "ccc", "tokenized_text": ["foo"], "ner": [], "source": "test"},
    ]
    holdout_ids = {"aaa", "ccc"}
    result = exact_dedup(train_rows, holdout_ids)
    assert len(result) == 1
    assert result[0]["sample_id"] == "bbb"


def test_exact_dedup_no_overlap():
    """When no overlap, all train rows are kept."""
    from prepare_data import exact_dedup

    train_rows = [
        {"sample_id": "xxx", "tokenized_text": ["a"], "ner": [], "source": "test"},
    ]
    holdout_ids = {"yyy", "zzz"}
    result = exact_dedup(train_rows, holdout_ids)
    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -m pytest tests/test_prepare_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'prepare_data'`

- [ ] **Step 3: Implement exact_dedup**

```python
# prepare_data.py
"""
Prepare contamination-free training data for GLiNER fine-tuning.

Loads arthrod/gliner-flex-pii-ready-v6-clean train split, removes any rows
that appear in the frozen holdout (arthrod/pii-gliner-evals) via:
  1. Exact sample_id match
  2. Fuzzy MinHash text similarity (Jaccard >= 0.85)

Outputs a clean parquet dataset ready for training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def exact_dedup(train_rows: list[dict], holdout_ids: set[str]) -> list[dict]:
    """Remove train rows whose sample_id appears in the holdout set."""
    return [r for r in train_rows if r["sample_id"] not in holdout_ids]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -m pytest tests/test_prepare_data.py::test_exact_dedup_removes_matching_sample_ids tests/test_prepare_data.py::test_exact_dedup_no_overlap -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/arthrod/workspace/hf-autoresearch
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add exact dedup for train vs holdout sample_ids"
```

---

### Task 2: Data Preparation Script — Fuzzy Dedup

**Files:**
- Modify: `prepare_data.py`
- Modify: `tests/test_prepare_data.py`

- [ ] **Step 1: Write failing test for fuzzy dedup**

```python
# tests/test_prepare_data.py (append)
def test_fuzzy_dedup_removes_near_duplicates():
    """Train rows with text highly similar to holdout text must be removed."""
    from prepare_data import fuzzy_dedup

    holdout_texts = [
        "Maria José da Silva mora na Rua das Flores número 42 em São Paulo",
    ]
    train_rows = [
        {
            "sample_id": "t1",
            "tokenized_text": "Maria José da Silva mora na Rua das Flores número 42 em São Paulo".split(),
            "ner": [],
            "source": "test",
        },
        {
            "sample_id": "t2",
            "tokenized_text": "O clima em Curitiba é frio no inverno".split(),
            "ner": [],
            "source": "test",
        },
    ]
    result = fuzzy_dedup(train_rows, holdout_texts, threshold=0.85)
    assert len(result) == 1
    assert result[0]["sample_id"] == "t2"


def test_fuzzy_dedup_keeps_dissimilar():
    """Dissimilar texts must be kept even if they share some tokens."""
    from prepare_data import fuzzy_dedup

    holdout_texts = ["Maria José da Silva mora na Rua das Flores"]
    train_rows = [
        {
            "sample_id": "t1",
            "tokenized_text": "Pedro Henrique Oliveira trabalha no Banco Central".split(),
            "ner": [],
            "source": "test",
        },
    ]
    result = fuzzy_dedup(train_rows, holdout_texts, threshold=0.85)
    assert len(result) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -m pytest tests/test_prepare_data.py::test_fuzzy_dedup_removes_near_duplicates -v`
Expected: FAIL — `ImportError: cannot import name 'fuzzy_dedup'`

- [ ] **Step 3: Implement fuzzy_dedup**

```python
# prepare_data.py (append after exact_dedup)
from datasketch import MinHash, MinHashLSH


def _text_to_shingles(text: str, n: int = 5) -> set[str]:
    """Convert text to character n-gram shingles."""
    return {text[i:i + n] for i in range(max(len(text) - n + 1, 1))}


def _make_minhash(shingles: set[str], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def fuzzy_dedup(
    train_rows: list[dict],
    holdout_texts: list[str],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> list[dict]:
    """Remove train rows whose text is similar to any holdout text (Jaccard >= threshold)."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # Index holdout texts
    for i, text in enumerate(holdout_texts):
        shingles = _text_to_shingles(text)
        mh = _make_minhash(shingles, num_perm)
        lsh.insert(f"holdout_{i}", mh)

    # Query each train row
    kept = []
    for row in train_rows:
        text = " ".join(row["tokenized_text"])
        shingles = _text_to_shingles(text)
        mh = _make_minhash(shingles, num_perm)
        matches = lsh.query(mh)
        if not matches:
            kept.append(row)

    return kept
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -m pytest tests/test_prepare_data.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/arthrod/workspace/hf-autoresearch
git add prepare_data.py tests/test_prepare_data.py
git commit -m "feat: add fuzzy MinHash dedup for train vs holdout"
```

---

### Task 3: Data Preparation Script — Full Pipeline with CLI

**Files:**
- Modify: `prepare_data.py`

- [ ] **Step 1: Write the full pipeline CLI**

```python
# prepare_data.py (append — main pipeline)
import argparse
import time

import datasets
from huggingface_hub import hf_hub_download


HOLDOUT_REPO = "arthrod/pii-gliner-evals"
TRAIN_REPO = "arthrod/gliner-flex-pii-ready-v6-clean"


def load_holdout() -> tuple[set[str], list[str]]:
    """Load frozen holdout. Returns (sample_ids, texts)."""
    path = hf_hub_download(HOLDOUT_REPO, "eval.json", repo_type="dataset")
    data = json.load(open(path))
    ids = {r["sample_id"] for r in data}
    texts = [" ".join(r["tokenized_text"]) for r in data]
    return ids, texts


def load_train() -> list[dict]:
    """Load v6-clean train split as list of dicts."""
    ds = datasets.load_dataset(TRAIN_REPO, split="train")
    rows = []
    for r in ds:
        rows.append({
            "tokenized_text": list(r["tokenized_text"]),
            "ner": list(r["ner"]),
            "source": r.get("source", "unknown"),
            "sample_id": r["sample_id"],
            "is_negative": r.get("is_negative", False),
            "ner_negatives": r.get("ner_negatives", []),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare contamination-free training data")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85)
    parser.add_argument("--skip-fuzzy", action="store_true", help="Skip fuzzy dedup (exact only)")
    args = parser.parse_args()

    print(f"Loading holdout from {HOLDOUT_REPO}...")
    holdout_ids, holdout_texts = load_holdout()
    print(f"  Holdout: {len(holdout_ids)} samples")

    print(f"Loading train from {TRAIN_REPO}...")
    train_rows = load_train()
    print(f"  Train: {len(train_rows)} samples")

    # Stage 1: exact dedup
    t0 = time.time()
    after_exact = exact_dedup(train_rows, holdout_ids)
    exact_removed = len(train_rows) - len(after_exact)
    print(f"  Exact dedup: removed {exact_removed}, kept {len(after_exact)} ({time.time()-t0:.1f}s)")

    # Stage 2: fuzzy dedup
    if not args.skip_fuzzy:
        t0 = time.time()
        after_fuzzy = fuzzy_dedup(after_exact, holdout_texts, threshold=args.fuzzy_threshold)
        fuzzy_removed = len(after_exact) - len(after_fuzzy)
        print(f"  Fuzzy dedup: removed {fuzzy_removed}, kept {len(after_fuzzy)} ({time.time()-t0:.1f}s)")
        final = after_fuzzy
    else:
        final = after_exact

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in final:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(final)} rows to {output_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"  Original train: {len(train_rows)}")
    print(f"  Exact removed:  {exact_removed}")
    if not args.skip_fuzzy:
        print(f"  Fuzzy removed:  {fuzzy_removed}")
    print(f"  Final clean:    {len(final)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify imports work**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -c "from prepare_data import main; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /home/arthrod/workspace/hf-autoresearch
git add prepare_data.py
git commit -m "feat: add full data preparation CLI with exact+fuzzy dedup"
```

---

### Task 4: Evaluation Script

**Files:**
- Create: `evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_evaluate.py
def test_evaluate_returns_strict_f1():
    """evaluate_predictions must return a dict with strict f1."""
    from evaluate import evaluate_predictions

    gold = [[{"label": "first name", "start": 0, "end": 5}]]
    pred = [[{"label": "first name", "start": 0, "end": 5, "text": "Maria", "score": 0.9}]]
    tags = ["first name"]
    result = evaluate_predictions(gold, pred, tags)
    assert "strict" in result
    assert result["strict"]["f1"] == 1.0


def test_evaluate_zero_on_mismatch():
    """Completely wrong predictions should score near zero."""
    from evaluate import evaluate_predictions

    gold = [[{"label": "first name", "start": 0, "end": 5}]]
    pred = [[{"label": "last name", "start": 10, "end": 20, "text": "wrong", "score": 0.9}]]
    tags = ["first name", "last name"]
    result = evaluate_predictions(gold, pred, tags)
    assert result["strict"]["f1"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement evaluate.py**

```python
# evaluate.py
"""
Evaluate a GLiNER checkpoint against the frozen holdout using vanilla Python GLiNER.

Usage:
    python evaluate.py --model-id arthrod/gliner-pii-mmbert-base-token-v1.0 --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nervaluate import Evaluator


HOLDOUT_REPO = "arthrod/pii-gliner-evals"

LABELS = [
    "cpf document number", "credit card", "dob", "email address", "first name",
    "last name", "location building number", "location city", "location full address",
    "location neighborhood", "location state", "location state abbreviation",
    "location street", "location zip", "middle name", "phone number",
    "pis document number", "race or ethnicity", "rg document number",
    "subject described medical data", "subject described organization affiliation",
    "subject described political opinion", "subject described religious conviction",
    "subject described sexual data",
]


def evaluate_predictions(
    gold: list[list[dict]],
    pred: list[list[dict]],
    tags: list[str],
) -> dict[str, dict]:
    """Run nervaluate and return {strategy: {precision, recall, f1}}."""
    results = Evaluator(gold, pred, tags=tags).evaluate()["overall"]
    return {
        s: {"precision": results[s].precision, "recall": results[s].recall, "f1": results[s].f1}
        for s in ["strict", "exact", "partial", "ent_type"]
    }


def load_holdout_gold() -> tuple[list[str], list[list[dict]]]:
    """Load holdout and build gold spans with char offsets. Returns (texts, gold_spans)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HOLDOUT_REPO, "eval.json", repo_type="dataset")
    data = json.load(open(path))

    texts, gold = [], []
    label_set = set(LABELS)
    for row in data:
        tokens = list(row["tokenized_text"])
        text = " ".join(tokens)
        texts.append(text)

        offsets, cursor = [], 0
        for t in tokens:
            offsets.append((cursor, cursor + len(t)))
            cursor += len(t) + 1

        spans = []
        for ann in row["ner"]:
            s, e, label = int(ann[0]), int(ann[1]), str(ann[2])
            if label not in label_set:
                continue
            if s < 0 or e >= len(offsets) or e < s:
                continue
            spans.append({"start": offsets[s][0], "end": offsets[e][1], "label": label})
        gold.append(spans)

    return texts, gold


def run_vanilla_inference(model_id: str, texts: list[str], threshold: float, batch_size: int = 4) -> list[list[dict]]:
    """Run vanilla GLiNER inference. Returns list of prediction lists."""
    import torch
    from gliner import GLiNER

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(model_id, map_location=device)

    all_preds = []
    for start in range(0, len(texts), 256):
        end = min(start + 256, len(texts))
        chunk_preds = model.inference(
            texts[start:end], LABELS, flat_ner=True,
            threshold=threshold, batch_size=batch_size,
        )
        all_preds.extend(chunk_preds)
        print(f"  Inference: {end}/{len(texts)}")

    return all_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLiNER checkpoint on frozen holdout")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    print(f"Loading holdout gold from {HOLDOUT_REPO}...")
    texts, gold = load_holdout_gold()
    print(f"  {len(texts)} samples, {sum(len(g) for g in gold)} gold entities")

    print(f"Running vanilla inference: {args.model_id} @ {args.threshold}...")
    t0 = time.time()
    preds_raw = run_vanilla_inference(args.model_id, texts, args.threshold, args.batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.1f} rows/s)")

    # Convert to nervaluate format
    pred_eval = []
    for row_preds in preds_raw:
        pred_eval.append([
            {"label": p["label"], "start": p["start"], "end": p["end"]}
            for p in row_preds if p["end"] > p["start"]
        ])

    gold_eval = [[{"label": s["label"], "start": s["start"], "end": s["end"]} for s in spans] for spans in gold]

    print("Running nervaluate...")
    results = evaluate_predictions(gold_eval, pred_eval, tags=LABELS)

    print(f"\n{'Strategy':<12} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 38)
    for s in ["strict", "exact", "partial", "ent_type"]:
        r = results[s]
        print(f"{s:<12} {r['precision']:8.4f} {r['recall']:8.4f} {r['f1']:8.4f}")

    if args.output:
        output = {
            "model_id": args.model_id,
            "threshold": args.threshold,
            "holdout_repo": HOLDOUT_REPO,
            "n_samples": len(texts),
            "inference_seconds": round(elapsed, 1),
            "strategies": results,
        }
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(f"\nSaved to {args.output}")

    return results


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -m pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/arthrod/workspace/hf-autoresearch
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation script using vanilla GLiNER + nervaluate"
```

---

### Task 5: Training Script

**Files:**
- Replace: `train.py`

- [ ] **Step 1: Write the GLiNER training script**

This replaces the GPT pretraining script entirely. It's a self-contained HF Jobs script that:
- Loads clean training data from a mounted bucket
- Fine-tunes GLiNER with the config from `d.yaml`
- Saves checkpoints to the results bucket

```python
# train.py
"""
GLiNER fine-tuning script for HF Jobs.

Usage (HF Jobs):
    hf jobs uv run \
        --flavor a100-large \
        --timeout 120m \
        --namespace arthrod \
        -v hf://buckets/arthrod/gliner-retrain-data:/data \
        -v hf://buckets/arthrod/gliner-retrain-results:/results \
        train.py
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gliner>=0.2.0",
#     "torch>=2.6.0",
#     "transformers>=4.48.0",
#     "datasets>=3.0.0",
#     "huggingface_hub>=0.25.0",
#     "wandb>=0.18.0",
# ]
# ///

import json
import os
import time
from pathlib import Path

import torch
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding

# ---------------------------------------------------------------------------
# Paths: auto-detect HF Jobs mounts vs local
# ---------------------------------------------------------------------------
DATA_DIR = "/data" if os.path.isdir("/data") else "./data"
RESULTS_DIR = "/results" if os.path.isdir("/results") else "./results"
TRAIN_FILE = os.path.join(DATA_DIR, "train_clean.jsonl")

# ---------------------------------------------------------------------------
# Hyperparameters (the agent modifies these)
# ---------------------------------------------------------------------------
BASE_MODEL = "jhu-clsp/mmBERT-base"  # or a checkpoint path
SPAN_MODE = "token_level"
MAX_LEN = 2048
NUM_STEPS = 40000
TRAIN_BATCH_SIZE = 12
EVAL_BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
LR_ENCODER = 1e-5
LR_OTHERS = 5e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
DROPOUT = 0.30
EVAL_EVERY = 2000
SAVE_LIMIT = 8
SEED = 42

# ---------------------------------------------------------------------------
# Setup and train
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    torch.manual_seed(SEED)

    # Load training data
    print(f"Loading training data from {TRAIN_FILE}...")
    train_data = [json.loads(l) for l in open(TRAIN_FILE) if l.strip()]
    print(f"  {len(train_data)} training samples")

    # Initialize model from config (not from pretrained GLiNER)
    print(f"Building GLiNER from backbone: {BASE_MODEL}")
    model = GLiNER.from_config(
        model_name=BASE_MODEL,
        span_mode=SPAN_MODE,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        max_neg_type_ratio=2,
        max_types=100,
    )

    output_dir = os.path.join(RESULTS_DIR, "checkpoint")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR_OTHERS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=EVAL_EVERY,
        save_steps=EVAL_EVERY,
        save_total_limit=SAVE_LIMIT,
        seed=SEED,
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    print(f"Starting training ({NUM_STEPS} steps)...")
    trainer.train()

    # Save final model
    final_path = os.path.join(RESULTS_DIR, "final_model")
    model.save_pretrained(final_path)
    print(f"Saved final model to {final_path}")

    elapsed = time.time() - t0
    print(f"\n---")
    print(f"training_seconds: {elapsed:.1f}")
    print(f"num_steps: {NUM_STEPS}")
    print(f"train_samples: {len(train_data)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -c "import ast; ast.parse(open('train.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /home/arthrod/workspace/hf-autoresearch
git add train.py
git commit -m "feat: replace GPT pretraining with GLiNER fine-tuning script"
```

---

### Task 6: Agent Instructions (program.md)

**Files:**
- Replace: `program.md`

- [ ] **Step 1: Write updated agent instructions**

```markdown
# GLiNER PII Retraining — Agent Instructions

This is an autonomous research loop for improving GLiNER PII detection models.

## Setup

1. **Agree on a run tag** with the user (e.g. `mar30`). Create branch `retrain/<tag>`.
2. **Create results bucket**: `hf buckets create arthrod/gliner-retrain-results`
3. **Prepare data**: Run `prepare_data.py` to build contamination-free training JSONL:
   ```bash
   python prepare_data.py --output /path/to/train_clean.jsonl
   ```
   Upload to data bucket: `hf buckets cp train_clean.jsonl hf://buckets/arthrod/gliner-retrain-data/train_clean.jsonl`
4. **Establish baseline**: Run `evaluate.py` on the current best model:
   ```bash
   python evaluate.py --model-id arthrod/gliner-pii-mmbert-base-token-v1.0 --threshold 0.5 --output baseline.json
   ```
5. **Initialize results.tsv** with the baseline.

## Running on HF Jobs

```bash
hf jobs uv run \
    --flavor a100-large \
    --timeout 120m \
    --namespace arthrod \
    -v hf://buckets/arthrod/gliner-retrain-data:/data \
    -v hf://buckets/arthrod/gliner-retrain-results:/results \
    train.py 2>&1 | tee run.log
```

## Evaluation

After each training run, evaluate the checkpoint using **vanilla Python GLiNER**:

```bash
python evaluate.py \
    --model-id /path/to/checkpoint \
    --threshold 0.5 \
    --output eval_result.json
```

**The metric is nervaluate strict F1 on the frozen holdout** (`arthrod/pii-gliner-evals`, 49,952 rows). Lower thresholds (0.3) and higher thresholds (0.7) can also be tested.

## What you CAN modify

- `train.py` — everything: hyperparameters, optimizer, scheduler, dropout, batch size, model architecture choices, loss function, data augmentation
- Thresholds in `evaluate.py` calls

## What you CANNOT modify

- `evaluate.py` — the evaluation harness is the ground truth
- `prepare_data.py` — the dedup pipeline is fixed
- The frozen holdout (`arthrod/pii-gliner-evals`)
- The label set (24 PII labels)

## Logging results

Record each experiment in `results.tsv` (tab-separated):

```
commit	strict_f1	exact_f1	memory_gb	status	paper	description
```

## The experiment loop

LOOP FOREVER:

1. **Research**: Use `hf papers search "NER"`, `"GLiNER training"`, `"token classification"`, etc.
2. **Implement**: Modify `train.py` with the idea.
3. **Commit**: `git commit` the change.
4. **Run**: Submit HF Job. Wait for completion.
5. **Evaluate**: Run `evaluate.py` on the checkpoint. Extract strict F1.
6. **Log**: Record in results.tsv.
7. If strict F1 improved → keep, save best to bucket.
8. If strict F1 equal/worse → `git reset` back.

**NEVER STOP.** The human will interrupt when done.
```

- [ ] **Step 2: Commit**

```bash
cd /home/arthrod/workspace/hf-autoresearch
git add program.md
git commit -m "feat: update agent instructions for GLiNER retraining loop"
```

---

### Task 7: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Replace the README with documentation for the GLiNER retraining pipeline, including:
- Overview (what this repo does now)
- Quick start (prepare data → establish baseline → run agent)
- Data contamination prevention (how dedup works)
- File descriptions
- Link to frozen holdout repo

- [ ] **Step 2: Commit**

```bash
cd /home/arthrod/workspace/hf-autoresearch
git add README.md
git commit -m "docs: update README for GLiNER retraining pipeline"
```

---

### Task 8: Integration Test — Dry Run

**Files:** None (verification only)

- [ ] **Step 1: Run prepare_data.py with --skip-fuzzy on a tiny subset**

This verifies the pipeline end-to-end without the expensive fuzzy dedup:

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -c "
from prepare_data import load_holdout, exact_dedup
holdout_ids, _ = load_holdout()
print(f'Holdout IDs: {len(holdout_ids)}')
# Simulate with 10 fake train rows, 2 overlapping
train = [{'sample_id': list(holdout_ids)[0], 'tokenized_text': ['test'], 'ner': [], 'source': 'test'},
         {'sample_id': 'unique_123', 'tokenized_text': ['clean'], 'ner': [], 'source': 'test'}]
result = exact_dedup(train, holdout_ids)
assert len(result) == 1
assert result[0]['sample_id'] == 'unique_123'
print('Integration test PASSED')
"`
Expected: `Integration test PASSED`

- [ ] **Step 2: Run evaluate.py in dry mode**

Run: `cd /home/arthrod/workspace/hf-autoresearch && python -c "
from evaluate import load_holdout_gold, evaluate_predictions, LABELS
texts, gold = load_holdout_gold()
print(f'Loaded {len(texts)} holdout samples, {sum(len(g) for g in gold)} gold entities')
# Verify gold format
assert len(gold) == len(texts)
assert all(isinstance(g, list) for g in gold)
print('Holdout gold loading PASSED')
"`
Expected: `Holdout gold loading PASSED`

- [ ] **Step 3: Commit all tests passing**

```bash
cd /home/arthrod/workspace/hf-autoresearch
python -m pytest tests/ -v
git add -A
git commit -m "test: verify full pipeline integration"
```
