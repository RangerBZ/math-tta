# Math TTA starter

A small, reproducible baseline for studying self-training and test-time
adaptation (TTA) on mathematical reasoning. It uses Hugging Face Transformers
for optional model generation and keeps data selection/evaluation independent
of the model backend.

## Current baseline

1. Generate multiple candidate solutions for each problem.
2. Extract final answers and select a candidate when its answer has sufficient
   agreement among the samples. The agreement fraction is its initial
   trajectory-value proxy.
3. Evaluate candidate accuracy, selected accuracy, and selection coverage
   against answers reserved for evaluation.

The selection step does **not** read reference answers. This is a baseline,
not yet the proposed learned/nested adaptation method: there is no parameter
update or learned outer-loop policy in this initial version.

## Setup

Use a dedicated environment. Install PyTorch separately using the command
appropriate for your CPU/CUDA setup, then install this project and its optional
generation dependencies:

```powershell
cd E:\Codes\Python\FYP\math-tta
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e ".[model]"
```

For a CUDA machine, use the PyTorch install selector at
<https://pytorch.org/get-started/locally/> rather than assuming the default
`pip install torch` build matches your driver.

Run the tests without downloading a model:

```powershell
python -m unittest discover -s tests -v
```

## Data format

Input problems are UTF-8 JSONL, one object per line:

```json
{"id":"example-1","question":"What is 12 times 13?","answer":"156"}
```

`answer` is optional and used only by `evaluate`. Keep it out of inputs used
for unlabeled test-time adaptation.

## Run

Generate `k` sampled solutions per prompt (the model is downloaded from the
Hugging Face Hub on first use unless already cached):

```powershell
math-tta generate --input data\sample.jsonl --output outputs\candidates.jsonl `
  --model Qwen/Qwen2.5-Math-1.5B-Instruct --samples 8 --max-new-tokens 512 --seed 42
```

Select agreement-based pseudo-labels. Only problems with a unique winning
answer and sufficient vote fraction are emitted:

```powershell
math-tta select --input outputs\candidates.jsonl `
  --output outputs\selected.jsonl --min-value 0.5
```

Evaluate generation and selection against reference answers:

```powershell
math-tta evaluate --problems data\sample.jsonl `
  --candidates outputs\candidates.jsonl --min-value 0.5
```

The `--model` value is configurable: use a model compatible with the available
hardware and tokenizer chat-template. Generation currently runs one problem
at a time to keep memory use predictable; increase throughput only after
measuring the baseline. A fixed seed helps reproduce sampling, though exact
reproducibility can still depend on hardware and library versions.

## Research progression

Keep the initial comparison controlled:

- zero-shot/single-sample generation;
- self-consistency with majority answer selection;
- this confidence-thresholded pseudo-label baseline;
- later, a parameter-efficient inner-loop update and a learned outer-loop
  trajectory-value/adaptation policy.

Track exact-match accuracy, selected-data coverage, update cost, and retained
performance on the source distribution. Tune thresholds on development data,
not on the final test benchmark.
