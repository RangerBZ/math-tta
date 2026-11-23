# Math TTA research prototype

A lightweight prototype for iterative, outcome-verified self-training on
mathematical reasoning. It can import GSM8K, sample model completions, verify
their extracted answers against labeled references, and fine-tune on accepted
traces using LoRA.

> **Scope note:** The prototype includes fixed-sample rejection-SFT, clipped
> GRPO with verifier or majority pseudo-label rewards, and the TTA-1 adaptive
> sampling policy described below. It does not implement adaptive trace
> weighting or step-level verification; answer verification checks only the
> final answer.

## Environment setup

Python 3.10 or later is required. Create an environment and install PyTorch
using the command that matches your machine (CPU or CUDA), then install this
project and its optional model-generation dependencies.

### Windows PowerShell

```powershell
cd E:\Codes\Python\FYP\math-tta
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the matching PyTorch build: https://pytorch.org/get-started/locally/
python -m pip install torch
python -m pip install -e ".[train]"
```

### Linux / SSH server using an existing CUDA environment

If your server provides a managed conda environment with a working PyTorch
CUDA build, use that environment directly. Do **not** create a separate venv
or install/upgrade PyTorch inside it. The project intentionally does not list
PyTorch as a dependency.

```bash
cd /path/to/math-tta
# If the shell prompt includes (.venv-math-tta), leave that venv first.
deactivate

# Activate the site-provided environment, for example:
conda activate llm

# Verify both Python and Torch come from the intended environment.
python -c 'import sys, torch; print("Python:", sys.executable); print("Torch:", torch.__file__); print("Versions:", torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'

# Install only the project and its Python training/data dependencies.
python -m pip install -e ".[train]"
```

Do not continue if the printed Python or Torch path contains
`.venv-math-tta`. The installation command does not select or replace the
PyTorch build; it uses whichever interpreter is active.

The optional dependencies pin Transformers to 4.57.3 (VeOmni's exact
requirement) and constrain Datasets to 2.16-2.21 to avoid incompatibilities
with a VeOmni-managed stack. PyTorch is intentionally not pinned in
`pyproject.toml`: its CUDA wheel must match the driver on the GPU node where the
job runs.

For example, CHPC `gpu035` reports driver 575.57.08 / CUDA 12.9, which cannot
initialize the installed `torch 2.15.0+cu132` build. The project works with
PyTorch 2.8.0+cu128 on a compatible node. In an environment you own and can
replace, install that wheel separately:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0"
python -m pip install -e ".[train]"
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The final command should report a `+cu128` build and `True` for CUDA
availability on an allocated GPU node. This replaces the existing PyTorch
installation, so do not run it in a managed/shared environment without
permission. Alternatively, keep the current `cu132` environment and run on a
node whose driver supports that build.

Test the model-independent components without downloading any model:

```bash
python -m unittest discover -s tests -v
```

The `train` extra installs Transformers, PEFT, Accelerate, and Hugging Face
Datasets. Install PyTorch separately using the command matching your CPU or
CUDA hardware. A model/dataset download is triggered the first time the
corresponding command is run. Check the model card and available hardware
before training:
[Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B).

## Prepare GSM8K

Import the official training split to JSONL:

```bash
math-tta prepare-gsm8k --split train --output data/gsm8k_train.jsonl
```

Each record contains `id`, `question`, and the GSM8K `answer` rationale, whose
final reference is extracted from its `####` marker. GSM8K has 7,473 training
questions and 1,319 held-out test questions. The test split can be exported for
final evaluation, but **must not** be passed to `train`.

## Run verified self-training

Start with a small pilot to verify model loading, answer extraction, and
hardware capacity:

```bash
math-tta train --input data/gsm8k_train.jsonl --output-dir outputs/qwen3-pilot --model Qwen/Qwen3-0.6B --samples 4 --limit-problems 128 --max-new-tokens 512 --rounds 1 --epochs 1 --batch-size 1 --max-length 2048 --seed 42
```

Omit `--limit-problems` to use the whole input file. Increase `--rounds` to
regenerate from the updated model and continue LoRA training on newly verified
traces. Training loads the base model in full precision unless CUDA supports
BF16; this avoids silently choosing an unsupported reduced-precision mode.
Candidate generation displays a tqdm bar with elapsed and estimated remaining
time, advancing once per completed problem. The training pipeline also flushes
candidate artifacts periodically during generation. Reduce batch/context sizes
if the available GPU memory is insufficient.

The pipeline records, for each round:

- `candidates-round-NN.jsonl`: generated completions and the exact generated
  token IDs, without reference answers.
- `verified-round-NN.jsonl`: extracted answers and verifier outcomes.
- `accepted-round-NN.jsonl`: only traces with a correct final answer.
- `metrics-round-NN.json`: parse rate, candidate accuracy, pass@k, and SFT loss.
- `round-NN/adapter/` and `final-adapter/`: PEFT LoRA adapter and tokenizer.

Use `--tune-mode full` to update all model parameters instead. Full checkpoints
are written under `round-NN/model/` and `final-model/`; this uses substantially
more GPU memory than the default LoRA mode.

## Run clipped GRPO

The `grpo` command uses the same labeled JSONL input and answer verifier, but
keeps both correct and incorrect sampled rollouts. It normalizes binary rewards
within each question's sample group and optimizes a clipped policy-ratio
objective. Sampling uses temperature 1, top-p 1, and top-k disabled so the
stored old-policy token probabilities match the sampling policy.

```bash
math-tta grpo \
  --input data/gsm8k_train.jsonl \
  --output-dir outputs/qwen3-grpo-pilot \
  --model Qwen/Qwen3-0.6B \
  --samples 4 \
  --limit-problems 128 \
  --max-new-tokens 512 \
  --rounds 1 \
  --epochs 1 \
  --batch-size 1 \
  --max-length 2048 \
  --learning-rate 1e-6 \
  --clip-epsilon 0.2 \
  --tune-mode lora \
  --seed 42
```

GRPO has no learned value critic. Groups in which every sample has the same
reward have zero group-relative advantage; the run reports and skips the update
if no group has mixed rewards. This initial implementation uses policy-ratio
clipping and gradient-norm clipping, but does not yet include a reference-policy
KL penalty, PPO's value critic/clipping, OPSD, or SDPO. It saves
`rollouts-round-NN.jsonl` with rewards, advantages, and old-policy token
log-probabilities, alongside verifier metrics and per-round checkpoints.
Sampling defaults to one GPU. Set `--sampling-devices 2` to replicate the model
across two visible CUDA GPUs and distribute question rollouts between them;
scoring and GRPO updates remain on the primary GPU. The generation batch size
applies independently to each replica.

For a two-H200 server, start with `--sampling-devices 2` and
`--generation-batch-size 1`; this parallelizes rollout generation, not training:

```bash
math-tta grpo \
  --input data/gsm8k_train.jsonl \
  --output-dir outputs/qwen3-grpo-h200 \
  --model Qwen/Qwen3-0.6B \
  --samples 4 \
  --sampling-devices 2 \
  --generation-batch-size 1 \
  --max-new-tokens 512 \
  --rounds 1 \
  --epochs 1 \
  --batch-size 1 \
  --max-length 2048 \
  --learning-rate 1e-6 \
  --clip-epsilon 0.2 \
  --tune-mode lora \
  --seed 42
```

### Majority pseudo-label reward mode

Use `--reward-mode pseudo-label` for a TTRL-style baseline. This mode extracts
and normalizes final answers from each question's rollouts, selects a unique
plurality answer only when its share meets `--min-consensus` (default `0.5`),
and rewards rollouts that match it. Tied or below-threshold groups are
abstentions with zero rewards. Reference answers are not read by this reward
path; `--input` may omit `answer` fields. Test-split IDs remain disallowed for
training to prevent evaluation leakage.

Run a small LoRA pilot from the repository root:

```bash
python -m math_tta.cli grpo \
  --input data/gsm8k_train.jsonl \
  --output-dir outputs/qwen3-pseudolabel-grpo \
  --model Qwen/Qwen3-0.6B \
  --reward-mode pseudo-label \
  --min-consensus 0.5 \
  --samples 4 \
  --limit-problems 128 \
  --max-new-tokens 512 \
  --rounds 1 \
  --epochs 1 \
  --batch-size 1 \
  --max-length 2048 \
  --learning-rate 1e-6 \
  --clip-epsilon 0.2 \
  --tune-mode lora \
  --seed 42
```

By default, the final LoRA checkpoint is
`outputs/qwen3-pseudolabel-grpo/final-adapter/`. Evaluate it on the held-out
GSM8K test split by loading the base model with that adapter, then scoring the
generated candidates:

```bash
python -m math_tta.cli generate \
  --input data/gsm8k_test.jsonl \
  --output outputs/qwen3-pseudolabel-grpo/test-candidates.jsonl \
  --model Qwen/Qwen3-0.6B \
  --adapter outputs/qwen3-pseudolabel-grpo/final-adapter \
  --samples 8 \
  --seed 42

python -m math_tta.cli evaluate \
  --problems data/gsm8k_test.jsonl \
  --candidates outputs/qwen3-pseudolabel-grpo/test-candidates.jsonl
```

To run full fine-tuning, replace `--tune-mode lora` with `--tune-mode full`.
The resulting `final-model/` is a standalone model checkpoint: pass that path
to `generate --model` and omit `--adapter`. Pseudo-label runs save
`pseudo-labels-round-NN.jsonl`, with the inferred answer, consensus, and
abstention status. Their metrics report pseudo-label coverage and consensus,
not accuracy against the unused training answers. The GRPO command uses tqdm
progress bars with elapsed and estimated remaining time for sampling, rollout
scoring, and each optimization epoch.

### TTA-1 adaptive sampling

TTA-1 is an adaptive rollout-budget policy for `grpo --reward-mode
pseudo-label`. It starts with `--samples` rollouts for every question, then
adds up to two samples at a time for unresolved questions. A group is accepted
only when its unique count-based answer winner reaches `--tta1-threshold`
(default `0.8`) and agrees with the confidence-weighted answer winner. It stops
sampling accepted groups; unresolved groups are sampled up to `--max-samples`
(default `8`) and then abstain. Unparseable rollouts count in the denominator
for count consensus but do not vote for an answer. Accepted groups reward only
rollouts matching the agreed count winner. Fixed sampling remains the default.

The confidence weight is derived from normalized mean next-token entropy over
each generated response. This is an uncalibrated self-certainty heuristic, not
a probability that the answer or reasoning is correct. The TTA-1 training
command does not use GSM8K reference answers:

```bash
python -m math_tta.cli grpo \
  --input data/gsm8k_train.jsonl \
  --output-dir outputs/qwen3-tta1-grpo \
  --model Qwen/Qwen3-0.6B \
  --reward-mode pseudo-label \
  --sampling-policy tta-1 \
  --samples 4 \
  --max-samples 8 \
  --tta1-threshold 0.8 \
  --max-new-tokens 512 \
  --rounds 1 \
  --epochs 1 \
  --batch-size 1 \
  --max-length 2048 \
  --learning-rate 1e-6 \
  --clip-epsilon 0.2 \
  --tune-mode lora \
  --seed 42
```

Round metrics include the mean number of samples per problem, how many groups
were extended, confidence-winner conflicts, abstentions, and pseudo-label
coverage. Because TTA-1 uses only rollout-derived signals, it can select a
wrong pseudo-label; evaluate on the held-out test set only after fixing the
policy and hyperparameters.

### TTA-2 confidence-shaped rewards

TTA-2 uses the same adaptive sampling, confidence/count-winner agreement, and
abstention decisions as TTA-1. It changes only reward allocation: for an
accepted group, a rollout matching the pseudo-label receives its saved
`token_self_certainty` in `[0, 1]`; dissenting or unaccepted rollouts receive
zero. This can create within-group GRPO advantages even when every sampled
answer agrees, provided their confidence scores differ. If all rewards in a
group are still equal, the group has zero relative advantage and is skipped.
Confidence is an uncalibrated self-certainty signal, not answer correctness;
TTA-2 can emphasize a confidently wrong pseudo-label.

Run TTA-2 with the same sampling controls as TTA-1:

```bash
python -m math_tta.cli grpo \
  --input data/gsm8k_train.jsonl \
  --output-dir outputs/qwen3-tta2-grpo \
  --model Qwen/Qwen3-0.6B \
  --reward-mode pseudo-label \
  --sampling-policy tta-2 \
  --samples 4 \
  --max-samples 8 \
  --tta1-threshold 0.8 \
  --max-new-tokens 512 \
  --rounds 1 \
  --epochs 1 \
  --batch-size 1 \
  --max-length 2048 \
  --learning-rate 1e-6 \
  --clip-epsilon 0.2 \
  --tune-mode lora \
  --seed 42
```

### Offline confidence/reward check

The rollout audit simulates TTA-2 rewards on the TTA-1 samples by default and
reports confidence-versus-training-answer accuracy bins alongside reward and
advantage distributions. This reuses the saved trajectories, so it checks the
reward mechanics without another sampling run or model update:

```bash
python -m math_tta.audit_rollouts \
  --problems data/gsm8k_train.jsonl \
  --fixed6-rollouts outputs/qwen3-fixed6-grpo/rollouts-round-01.jsonl \
  --tta1-rollouts outputs/qwen3-tta1-grpo/rollouts-round-01.jsonl \
  --initial-samples 4 \
  --fixed-min-consensus 0.5 \
  --tta1-threshold 0.8 \
  --output-dir outputs/rollout-audit
```

Optionally pass `--tta2-rollouts` with the actual TTA-2 rollout artifact to
check that recomputed TTA-2 rewards and advantages match the saved values.
The confidence/correctness relationship is diagnostic only: training answers
are not used by the TTA-2 reward.

### Audit fixed-6 and TTA-1 rollout groups

After both training runs, compare each generated answer with its GSM8K
training reference, inspect pseudo-label changes at the 4/6/8-rollout
prefixes, and summarize the resulting binary rewards and group-relative
advantages:

```bash
python -m math_tta.audit_rollouts \
  --problems data/gsm8k_train.jsonl \
  --fixed6-rollouts outputs/qwen3-fixed6-grpo/rollouts-round-01.jsonl \
  --tta1-rollouts outputs/qwen3-tta1-grpo/rollouts-round-01.jsonl \
  --initial-samples 4 \
  --fixed-min-consensus 0.5 \
  --tta1-threshold 0.8 \
  --output-dir outputs/rollout-audit
```

Pass the **raw** `rollouts-round-NN.jsonl` files, not only the `score` command's
verified output: adaptive-policy auditing needs each rollout's saved
`token_self_certainty` to reproduce its confidence-weighted decision. The
report writes per-question files and a `summary.json` with rollout correctness,
pseudo-label coverage/accuracy, label and reward changes, mixed-reward groups,
and zero/nonzero advantages. Prefix-cohort metrics include only groups that
reached that sample count; each prefix also reports cumulative policy state,
retaining groups that stopped earlier. Ground-truth answers are used only for
this offline audit; they do not affect either run's rewards.

Unparseable generations count as failed attempts. In verifier mode, answer
agreement does not decide which traces train the model; the reference-answer
verifier does. In pseudo-label mode, normalized within-group answer agreement
is the reward signal and can reinforce a wrong answer. LoRA checkpoints are
PEFT-format and require the original base model when loaded for inference.

Evaluate the final adapter by generating from the base model with that adapter
attached:

```bash
math-tta generate --input data/gsm8k_test.jsonl --output outputs/test_candidates.jsonl --model Qwen/Qwen3-0.6B --adapter outputs/qwen3-pilot/final-adapter --samples 8 --seed 42
math-tta evaluate --problems data/gsm8k_test.jsonl --candidates outputs/test_candidates.jsonl
```

Keep the test answers out of training, checkpoint selection, and sampling-policy
tuning; use them only for the final evaluation.

To score previously generated candidates without training:

```bash
math-tta score --problems data/gsm8k_train.jsonl --candidates outputs/candidates.jsonl --output outputs/verified.jsonl
```

SFT uses the exact sampled token IDs after the same prompt prefix used during
generation. This avoids reconstructing model-specific chat-template tokens
from decoded completion text. If an earlier run failed with “no tokenized
assistant targets,” rerun it with this version: the old candidate artifacts
did not include these token IDs. Sampled-token log-probabilities are optional
because they require an additional forward pass for each completion. Enable
`--record-token-logprobs` on `generate` or `train` to record the raw model
log-probability for each sampled token (rather than the
temperature/top-p-modified sampling distribution). Scoring reports the mean
when these values are present; they are confidence diagnostics and do not
affect answer verification or training selection.

## Run the bundled demonstration

The project includes two math problems and six prewritten candidate solutions.
Use them to exercise selection and evaluation without model downloads:

```bash
math-tta select --input data/sample_candidates.jsonl --output outputs/selected.jsonl --min-value 0.5
math-tta evaluate --problems data/sample.jsonl --candidates data/sample_candidates.jsonl --min-value 0.5
```

The evaluation prints:

```json
{
  "candidate_pass_at_k": 1.0,
  "candidate_accuracy": 0.5,
  "labeled_problems": 2,
  "min_value": 0.5,
  "parse_rate": 1.0,
  "problems_with_candidates": 2,
  "selected_accuracy": 0.5,
  "selected_count": 2,
  "selection_coverage": 1.0
}
```

Interpretation:

- `candidate_pass_at_k`: fraction of problems for which **at least one**
  sampled candidate has the correct final answer. It is a problem-level
  pass@k, not the fraction of individual candidate strings that are correct.
- `selected_accuracy`: fraction of selected problems whose majority-vote
  answer is correct.
- `selection_coverage`: fraction of problems with candidates for which the
  selection rule emitted a pseudo-label.

The second demonstration problem intentionally has two candidates answering
`155` and one answering `156`; the gold answer is `156`. Thus, a correct
candidate exists, but the majority selector chooses incorrectly. Agreement is
a confidence proxy, not a correctness guarantee.

## Generate candidates with Qwen3-0.6B

The input is UTF-8 JSONL with one question per record. `answer` may be present
for later evaluation but is not read by candidate generation or selection:

```json
{"id":"problem-1","question":"What is 12 times 13? Give the final answer in a boxed expression.","answer":"156"}
```

Sample eight candidate completions for each input problem:

```bash
math-tta generate --input data/sample.jsonl --output outputs/candidates.jsonl --model Qwen/Qwen3-0.6B --samples 8 --max-new-tokens 1024 --temperature 0.6 --top-p 0.95 --seed 42
```

Then select and measure the candidates:

```bash
math-tta select --input outputs/candidates.jsonl --output outputs/selected.jsonl --min-value 0.5
math-tta evaluate --problems data/sample.jsonl --candidates outputs/candidates.jsonl --min-value 0.5
```

`--min-value` is the minimum winning-answer vote fraction. Selection also
requires a unique majority; tied votes and groups without extractable final
answers are not selected. The fixed seed helps reproduce sampling, but exact results may vary with
hardware and software versions. Increase `--batch-size` to batch prompts
together when memory permits.

## Research experiment cautions

For research results, keep separate training, development, and held-out test
data. Use training answers for outcome verification and training only. Do not
tune prompts, sampling policies, or checkpoints using the held-out test set.
Match generation and training compute when comparing adaptive policies with
fixed-sample rejection-SFT. The bundled toy problems are demonstrations, not
evaluation data.