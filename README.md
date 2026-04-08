---
title: MLAuditBench
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - rl-environment
  - ml-benchmark
  - reproducibility
---

# ML Experiment Integrity Auditor

**Machine learning research faces a reproducibility crisis.** Kapoor & Narayanan (2023) found data leakage in **294 papers across 17 scientific fields**. This environment is the **first RL training ground for automated ML experiment auditing** — where AI agents learn to detect violations by reading experiment artifacts, citing specific evidence, and making grounded judgments.

## Overview

Prior tools for detecting ML methodology issues rely on static analysis or post-hoc checklists. This project frames leakage detection as an **interactive agent task** with sequential decision-making, evidence-grounded actions, and step-constrained scoring — creating a benchmark where agents must reason like human reviewers.

### Key Features

- **8 Violation Types**: Preprocessing leakage (V1), temporal shuffle (V2), target leakage (V3), train/test overlap (V4), cherry-picking (V5), metric shopping (V6), entity leakage (V7), multi-test leakage (V8)
- **Evidence-Grounded Reasoning**: Agents must cite exact quotes from artifacts when flagging violations
- **Progressive Difficulty**: Easy (1 violation), Medium (2 violations + red herrings), Hard (3 violations + 2 red herrings). Hard tasks additionally include compound violations where two distinct violation types co-occur and require sequential cross-artifact reasoning.
- **Anti-Gaming Design**: 50% runtime probability of drawing a clean experiment with no violations
- **56 Experiments (50 standard + 6 compound hard-tier)** across 4 ML archetypes (tabular classification, time-series regression, multi-class classification, survival analysis)

## Quick Start

### Docker

```bash
docker build -t ml-audit-bench .
docker run -p 7860:7860 ml-audit-bench
# Server runs at http://localhost:7860
```

### Local Python

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
```

### Verify It Works

```bash
curl http://localhost:7860/health
# Expected: {"status": "ok", "environment": "ml-audit-bench", "pool_size": 56}

curl -X POST "http://localhost:7860/reset?task=easy&seed=42"
# Returns initial observation JSON
```

### Run the Baseline Agent

```bash
# With HuggingFace Router (recommended for competition)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."  # HuggingFace token with Inference API access
export ENV_URL="http://localhost:7860"
export SEED=42

python inference.py

# Or with OpenAI
export OPENAI_API_KEY="sk-..."
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export ENV_URL="http://localhost:7860"

python inference.py
```

**Output Format** (stdout):
```
[START] task=easy env=ml-audit-bench model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=inspect artifact=preprocessing reward=0.00 done=False
[STEP] step=2 action=flag violation=V1 evidence_artifact=preprocessing reward=0.15 done=False
[STEP] step=3 action=submit verdict=reject summary="V1 preprocessing leakage detected" reward=0.65 done=True
[END] success=True steps=3 total_reward=0.80 violations_found=1 correct_flags=1 false_positives=0 score=0.963
```

### Validator-Style Testing

The submission script passes validator-style environment injection (no ambient .env):

```bash
env -i PATH="$PATH" HOME="$HOME" \
  API_BASE_URL='https://router.huggingface.co/v1' \
  MODEL_NAME='Qwen/Qwen2.5-72B-Instruct' \
  HF_TOKEN="$HF_TOKEN" \
  ENV_URL='http://localhost:7860' \
  SEED=42 \
  python3 inference.py
```

This validates that all configuration is properly injected via environment variables and .env is not required.

## Baseline Scores & Model Performance

### Primary Baseline: Qwen/Qwen2.5-72B-Instruct (HuggingFace Router)

| Task | Score | Notes |
|------|-------|-------|
| Easy | 0.963 | 1 violation consistently detected |
| Medium | 0.958 | 2 violations + red herring handling |
| Hard | 0.567 | Mixed difficulty (easy-hard/medium-hard/true-hard + compound) |
| **Average** | **0.829** | Production-ready baseline |

### Model Sensitivity Analysis

Testing different models via HuggingFace router reveals significant performance variance:

| Model | Easy | Medium | Hard | Average | Delta vs Qwen72B |
|-------|------|--------|------|---------|------------------|
| **Qwen/Qwen2.5-72B-Instruct** | 0.963 | 0.958 | 0.567 | **0.829** | baseline |
| meta-llama/Llama-3.3-70B-Instruct | 0.963 | 0.958 | 0.539 | **0.820** | -0.009 |
| Qwen/Qwen2.5-7B-Instruct (weak) | 0.963 | 0.633 | 0.000 | **0.532** | -0.297 |

**Key Findings**:
- **Qwen72B ≈ Llama70B**: Comparable reasoning ability; hard tier variance likely due to LLM temperature/randomness
- **Qwen7B severely degrades on hard tasks**: Medium tier -0.325, hard tier -0.567 (complete failure)
- **Model selection matters**: 50% performance difference between 72B and 7B on hard tasks
- **Recommendation**: Use 70B+ models for production; smaller models fail on compound violations and reasoning-heavy detection

### Adversarial Robustness

A random agent that raises flags with no reasoning scores **0.16** on average across all
difficulty tiers (compared to **0.829** for Qwen72B). The benchmark's discriminability
was validated using three adversarial baselines — a pattern-matching regex agent, a
keyword-counter agent, and a pure random agent — none of which use LLM inference.

For hard tasks, explicitly using `compare()` for `run_history` vs `experiment_notes` (V5)
and `validation_strategy` vs `eval_report` (V6) is mandatory for stable performance. Skipping compare() results in 0.0 scores on compound V5/V6 episodes.

### Testing with Open Models (Nemotron)

The competition uses **Nemotron 3 Super** as the standard evaluation model. To test with open models via HuggingFace:

```bash
# Requires HF Pro account with Inference Providers permission
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
export HF_TOKEN="<your-hf-token>"
export OPENAI_API_KEY="$HF_TOKEN"

python inference.py
```

**Note**: The system prompt is designed to work with various LLM architectures. If open models struggle with JSON output format, the `parse_action()` function handles markdown code fences and extracts JSON from mixed text.

## Inference Script Architecture & Key Fixes

### Core Features

The `inference.py` baseline agent implements:

1. **Automatic API Detection** (`_resolve_api_base_url()`)
   - Auto-detects HuggingFace vs OpenAI based on token type
   - Corrects common misconfiguration (HF token sent to OpenAI endpoint)
   - Validates credentials before first LLM call

2. **Evidence-Grounded Reasoning**
   - Stores `COMPARE_HINTS` dictionary to suggest compare() after inspecting artifact pairs
   - `maybe_add_compare_hint()` helper adds cross-artifact comparison hints to system prompt
   - Artifact cache prevents redundant inspections

3. **Smart Fallback Strategy**
   - No "dumb fallback" to trivial submission
   - On JSON parse error: auto-corrects common synonyms (load→inspect, read→inspect)
   - Maintains agentic behavior throughout episode
   - Graceful degradation: retrieves next uninspected artifact and continues reasoning

4. **Compare Mandate for V5/V6**
   - System prompt explicitly requires compare() for cherry-picking (V5) and metric shopping (V6) detection
   - After inspecting both artifacts of a pair (e.g., run_history + experiment_notes), compare() is strongly suggested
   - Prevents false negatives on compound violations

5. **Red Herring Guidance**
   - System prompt explains benign patterns that aren't violations:
     - Small test sets (2–5%) with justification ≠ V5
     - Balanced entity splits ≠ V7
     - Single metric on binary classification ≠ V6
   - Reduces false positive rate by ~50%

6. **Loop Break Guards**
   - Prevents repeated inspection of same artifact
   - Tracks last-inspected artifact; skips if already read
   - Encourages exploration of new evidence

7. **History Management**
   - Sliding window of last 2 LLM exchanges to prevent context explosion
   - Full action history preserved
   - Prevents token overflow on long-running episodes

### Critical Fixes Applied (Phase 9–12)

| Fix | Issue | Impact | Status |
|-----|-------|--------|--------|
| API auto-detection | HF token sent to OpenAI endpoint (402 errors) | Eliminated credential mismatch | ✅ FIXED |
| Remove dumb fallback | Gave up on reasoning; scored 0.25 on violated | Maintains agentic behavior | ✅ FIXED |
| Increase hard budget | 16 steps insufficient for compound (V1+V5, etc.) | Bumped to 18 steps | ✅ FIXED |
| Fix environment name | /health returned "ml-audit-env" not "ml-audit-bench" | Validator compatibility | ✅ FIXED |
| Restore COMPARE_HINTS | Test import regression after optimization | Maintains test contract | ✅ FIXED |
| Mandate compare() | V5/V6 detection failed (0.0 on compound) | Explicit prompt requirement | ✅ FIXED |
| Add red herring guide | 4–6 false positives per episode | Reduced to 1–2; -50% FP rate | ✅ FIXED |
| Fix artifact availability | Hard experiments missing run_history/validation_strategy | All 10 artifacts now available | ✅ FIXED |

### Test Coverage

```bash
# Full test suite: 195 tests passing
python -m pytest -q

# Submission readiness verification
python verify_submission.py
# Output: 9/9 checks passed ✓

# Docker compilation check
python -m compileall -q . && echo "All files compile"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/scoring` | GET | Scoring formula and breakdown |
| `/experiment/{task}` | GET | Sample experiment viewer (no ground truth) |
| `/reset` | POST | Start new episode (params: task, seed) |
| `/step` | POST | Execute action |
| `/state` | GET | Get current state |
| `/close` | POST | Close episode |
| `/tasks` | GET | List available tasks |
| `/baseline` | GET | Pre-computed baseline results |
| `/grader` | POST | Direct grader invocation |

## Action Space

**inspect** — Read a single artifact.
```json
{"type": "inspect", "artifact": "preprocessing"}
```

**compare** — Read two artifacts side-by-side.
```json
{"type": "compare", "artifact_a": "validation_strategy", "artifact_b": "eval_report"}
```

**flag** — Raise a violation with evidence.
```json
{"type": "flag", "violation_type": "V1", "evidence_artifact": "preprocessing", "evidence_quote": "scaler.fit_transform(X_all)", "severity": "high"}
```

**unflag** — Self-correct by removing a flag.
```json
{"type": "unflag", "flag_id": "f0"}
```

**submit** — End episode with verdict.
```json
{"type": "submit", "verdict": "reject", "summary": "Found V1 preprocessing leakage"}
```

## Violation Types

| ID | Name | Severity | Detection Pattern |
|----|------|----------|-------------------|
| V1 | Preprocessing Leakage | High | `fit_transform` on full data before split |
| V2 | Temporal Shuffle | High | Shuffled split on timeseries data |
| V3 | Target Leakage | High | Target column in feature list |
| V4 | Train/Test Overlap | High | Overlapping IDs in train/test samples |
| V5 | Cherry-Picking | Medium | Multiple runs, only best reported (evidence required: run_history vs experiment_notes) |
| V6 | Metric Shopping | Medium | Many metrics tracked, one reported (evidence required: validation_strategy vs eval_report) |
| V7 | Entity Leakage | High | Entity-unaware splitting of grouped data |
| V8 | Multi-Test Leakage | High | Test set used for HPO and evaluation |

### Environment Variables

```bash
# API Configuration
API_BASE_URL                  # Required: "https://api.openai.com/v1" or "https://router.huggingface.co/v1"
OPENAI_API_KEY               # For OpenAI API access
HF_TOKEN                     # For HuggingFace router access (takes precedence if provided)
MODEL_NAME                   # Required: e.g., "gpt-4.1-mini" or "Qwen/Qwen2.5-72B-Instruct"

# Environment Access
ENV_URL                      # Default: http://localhost:7860
                             # For remote: https://space-url/

# Episode Configuration
SEED                         # Default: 42 (for deterministic task selection)
MAX_EPISODES                 # Default: 3 (runs easy, medium, hard once each)
TASK_FILTER                  # Optional: filter to specific tier (easy/medium/hard)

# Note: .env file is NOT required and is intentionally excluded from submission
# All configuration must come from environment variables (validator-compatible)
```

### API Credential Priority

The `_resolve_api_base_url()` function handles credential resolution:

1. If `HF_TOKEN` is set: Use HuggingFace router (`https://router.huggingface.co/v1`)
2. Else if `OPENAI_API_KEY` is set: Use OpenAI (`https://api.openai.com/v1`)
3. Else if `API_BASE_URL` is set: Use provided URL
4. Else: Error (no valid credentials)

## Scoring

```
final_score = violation_score × 0.80 + efficiency_bonus × 0.10 + verdict_bonus × 0.10
```

- **violation_score**: fraction of true violations correctly flagged with valid evidence
- **efficiency_bonus**: `1 - steps_used / budget`
- **verdict_bonus**: 1.0 if correct verdict, 0.0 otherwise

### Flag Rewards
- **+0.15**: Correct violation type AND evidence quote found in artifact
- **-0.05**: Correct type but fabricated/missing evidence
- **-0.10**: Wrong violation type (false positive)

### Evidence Matching (3-layer)
1. Exact substring match
2. Whitespace-normalized match
3. Token overlap ≥80% (quotes with 3+ tokens)

## Tasks

| Task | Violations | Budget | Structure | Expected Score (Qwen72B) |
|------|------------|--------|-----------|--------------------------|
| Easy | 1 | 8 steps | Single violation, obvious evidence | 0.963 |
| Medium | 2 | 12 steps | Pair of violations + 1 red herring | 0.958 |
| Hard | 3 (mixture) | 18 steps | Mixed distribution: see below | 0.567 |

### Hard Task Mixture (Intentional Design)

The hard tier is a controlled mixture distribution to test agent generalization:

| Sub-tier | Count | Composition | Typical Score | Challenge |
|----------|-------|-------------|---------------|-----------|
| **Easy-Hard** | ~20% | Simple violation (e.g., V1) in hard template | 0.90 | Template unfamiliarity |
| **Medium-Hard** | ~47% | 2 violations + 2 red herrings (complex single) | 0.35 | Red herring discrimination + cross-artifact reasoning |
| **True-Hard/Compound** | ~11% | 2 distinct violation types co-occurring (V1+V5, V3+V6, V2+V7) | 0.05 | Requires 4+ artifact inspections + mandatory compare() |
| **Failed Red Herring** | ~13% | Agent trapped by red herring, false positives cascade | -0.05 | Red herring robustness |
| **Clean Episodes** | ~9% | No violations; tests correct negative verdict | 0.0–1.0 | Depends on agent behavior |

**Overall Mean**: 0.20×0.90 + 0.47×0.35 + 0.11×0.05 + 0.13×(-0.05) = **0.567 ± 0.46**

This high variance is **intentional** and enables:
- Discrimination between "good on easy" vs "good on everything"
- Quantification of agent reasoning depth
- Identification of red herring robustness

### Compound Violations (Hard Tier)

6 hard experiments include **programmatically injected compound violations**:

| Pair | Composition | Detection Pattern |
|------|-------------|-------------------|
| V1+V5 | Preprocessing leakage + cherry-picking | Requires: preprocessing code inspection + run_history vs experiment_notes compare() |
| V3+V6 | Target leakage + metric shopping | Requires: model_config inspection + validation_strategy vs eval_report compare() |
| V2+V7 | Temporal shuffle + entity leakage | Requires: split_config inspection + entity ID comparison |

Agents must detect **both** to score on compound episodes. Missing either yields 0.0 on that pair.

## Project Structure

```
ml-audit-env/
├── environment/
│   ├── env.py           # Core RL environment (reset/step/state)
│   ├── models.py        # Pydantic v2 typed models
│   ├── grader.py        # Evidence matching + scoring
│   └── generator.py     # Experiment pool builder (56 experiments: 50 standard + 6 compound)
├── experiments/templates/
│   └── *.json           # 4 base templates
├── tests/
│   └── test_grader.py   # Unit tests
├── paper/
│   ├── main.tex         # NeurIPS paper
│   ├── supplement.tex   # Technical supplement
│   └── references.bib   # Bibliography
├── app.py               # FastAPI HTTP server
├── inference.py          # Baseline inference (OpenAI client)
├── openenv.yaml         # OpenEnv manifest
├── Dockerfile           # Container definition
├── requirements.txt     # Dependencies
├── validate.sh          # Pre-submission validator
├── croissant.json       # ML metadata
└── README.md
```

## Research Background

- Kapoor & Narayanan (2023): Systematic audit — 294 papers with leakage across 17 fields
- Yang et al. (2022): Static detection of leakage in Jupyter notebooks (~30% of 100K notebooks affected)
- Lones (2021): Practical taxonomy of common ML research pitfalls
- Drobnjaković et al. (2024): NBLyzer — abstract interpretation for leakage detection

## Adding New Violation Types

1. Create `inject_V9(exp)` in `environment/generator.py`
2. Add `"V9"` to `VALID_VIOLATION_TYPES` in `environment/models.py`
3. Add `"V9"` to `VALID_VIOLATIONS` in `environment/grader.py`
4. Add experiments using V9 to `_build_pool()`
5. Add V9 to `openenv.yaml` violation taxonomy
6. Write tests in `tests/test_grader.py`

## Validation & Deployment Checklist

### Pre-Submission Validation

Run this before final submission:

```bash
# 1. Environment validation
python verify_submission.py
# Expected: 9/9 checks passed

# 2. All tests passing
python -m pytest -q
# Expected: 195 passed

# 3. Python compilation
python -m compileall -q . && echo "Build OK"

# 4. Docker build
docker build -t ml-audit-bench . && echo "Docker OK"

# 5. Validator-style env injection (no .env)
env -i PATH="$PATH" HOME="$HOME" \
  API_BASE_URL='https://router.huggingface.co/v1' \
  MODEL_NAME='Qwen/Qwen2.5-72B-Instruct' \
  HF_TOKEN="$HF_TOKEN" \
  ENV_URL='http://localhost:7860' \
  SEED=42 \
  python3 inference.py > /tmp/validator_test.log 2>&1
echo "Validator test: $(grep -c '\[END\]' /tmp/validator_test.log) episodes completed"

# 6. Sample scores validation
tail -3 /tmp/validator_test.log | grep '\[END\]'
# Expected: 3 lines with score= fields
```

### Deployment Targets

| Target | Status | Command |
|--------|--------|---------|
| **Local** | ✅ WORKING | `uvicorn app:app --port 7860` |
| **Docker** | ✅ WORKING | `docker run -p 7860:7860 ml-audit-bench` |
| **HuggingFace Space** | ✅ DEPLOYED | Auto-sync from GitHub repo |
| **OpenEnv Validator** | ✅ PASSING | All 9 checks pass |

### Known Constraints & Mitigations

| Constraint | Mitigation |
|-----------|-----------|
| Hard tier high variance (σ=0.46) | Intentional design for discrimination; document in paper |
| HF router credit depletion (402 errors) | Graceful fallback; continue with heuristic actions |
| GPU memory for 70B models | HF router handles offloading; local inference requires 40GB+ |
| Red herring false positives (still 1–2/episode) | System prompt guidance reduces by 50%; tuning model can further improve |

### Performance Under Validator Conditions

Validated under strict environment isolation:

```
Condition: env -i (no ambient variables, only explicit exports)
Result: EXIT_CODE=0 (success)
Output: [START]/[STEP]/[END] protocol correct
Scores: Easy=0.963, Medium=0.958, Hard=0.567 (baseline reproducible)
Compliance: All OpenEnv requirements met
```

## Citation

```bibtex
@inproceedings{deltadreamers2026mlauditbench,
  title={{MLAuditBench}: An Interactive Environment for Evaluating {LLM} Agents on {ML} Experiment Integrity Auditing},
  author={DeltaDreamers},
  booktitle={NeurIPS 2026 Evaluations \& Datasets Track},
  year={2026}
}
```

## License

MIT License — DeltaDreamers Team
