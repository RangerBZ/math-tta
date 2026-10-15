# Math TTA research prototype

A lightweight prototype for sampling multiple mathematical reasoning
trajectories, selecting pseudo-labels by answer agreement, and evaluating
selection. The planned research experiments use Qwen3-0.6B and GSM8K; the
current code is only the generation/selection/evaluation baseline.

> **Scope note:** This repository does not yet fine-tune the model or implement
> an adaptive update algorithm. It does not download or preprocess GSM8K. See
> the separate FYP workspace research report for the proposed method and
> experiment design.

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
python -m pip install -e ".[model]"
```

### Linux / SSH server

```bash
cd /path/to/math-tta
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the matching PyTorch build: https://pytorch.org/get-started/locally/
python -m pip install torch
python -m pip install -e ".[model]"
```

Test the model-independent components without downloading any model:

```bash
python -m unittest discover -s tests -v
```

For Qwen3, Transformers 4.51.0 or later is required. A model download is
triggered the first time the `generate` command is run. Check the model card
and available hardware before starting generation:
[Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B).

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
  "labeled_problems": 2,
  "min_value": 0.5,
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
answers are not selected. The fixed seed helps reproduce sampling, but exact
results may vary with hardware and software versions. Generation currently
processes one problem at a time.

## GSM8K experiment status

The commands above demonstrate only the current prototype. They do **not**
train Qwen3, build GSM8K JSONL files, or implement the proposed adaptive
sampling/weighting method. Do not evaluate a research result using the two
bundled toy problems. For the intended experiment, prepare separate training,
development, and held-out test data; use training answers only for training
selection/scoring, and keep test answers out of all adaptation and tuning.
