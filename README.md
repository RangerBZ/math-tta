# Math TTA research prototype

A lightweight prototype for iterative, outcome-verified self-training on
mathematical reasoning. It can import GSM8K, sample model completions, verify
their extracted answers against labeled references, and fine-tune on accepted
traces using LoRA.

> **Scope note:** This implements fixed-sample rejection-SFT as a baseline. It
> does not yet implement adaptive sampling/weighting or RLVR/GRPO. Answer
> verification checks only the final answer; it does not verify each reasoning
> step.

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
with a VeOmni-managed stack.
PyTorch 2.8.0+cu128 is compatible with the project's code and remains supplied
by the existing environment; the install command above does not request a
different PyTorch version. Avoid running `pip install torch` in this environment.

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
final reference is extracted from its `####` marker. The test split can be
exported for final evaluation, but **must not** be passed to `train`.

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
The CLI prints flushed progress during model loading, generation, and SFT, and
flushes candidate artifacts periodically during generation. Reduce batch/context
sizes if the available GPU memory is insufficient.

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

Unparseable generations count as failed attempts. Answer agreement does not
decide which traces train the model: only the reference-answer verifier does.
The saved adapter is PEFT-format and requires the original base model when
loaded for inference.

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
