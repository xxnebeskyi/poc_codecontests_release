# One-round self-training on CodeContests-O

Ministral 3B Reasoning generated two candidates per problem and scored its own code and explanation. One QLoRA adapter was trained on 38 selected responses. Held-out execution passes increased from 8/60 to 12/60, with four improvements and no regressions. The judge picked the passing candidate in four of the five held-out pairs where exactly one candidate passed.

## Inspect the results

Open `analysis.ipynb` for the saved tables and plots. To rerun it, start Jupyter from this directory with pandas and matplotlib installed. A GPU is only required for a new experiment.

- `results/paired_results.csv` contains the per-problem comparison. `original_0` is the paired baseline, `original_1` is the second baseline draw used for ranking, and `adapted` is the fine-tuned draw.
- `results/candidates.jsonl`, `judgments.jsonl`, and `execution.jsonl` join on `id`. The records contain model outputs, selection decisions, and execution outcomes. Training records with an accepted judgment identify the fine-tuning set.
- `results/manifest.json` records the prompts, revisions, and settings. `results/problems.json` contains the frozen split and test cases. `training-aececdb7.jsonl` contains the main adapter updates.

## Repeat the experiment

Use a CUDA-enabled PyTorch environment and Docker with Linux containers. Install the remaining runner dependencies, then launch from this directory.

```powershell
python -m pip install -r requirements.txt
python run.py --run-dir repeat --hours 9
```

The runner downloads the pinned Ministral checkpoint from Hugging Face and loads the recorded problems and test cases. It retains the pilot, selection, training, and evaluation stages. Use the same directory to resume a repeat. The original run used PyTorch 2.11.0+cu128 and the Docker image ID in the saved manifest. The repeat records the locally resolved `python:3.12-slim` image ID. Image and hardware differences can affect the outcome.

## Data provenance

Problem statements and cases are from [CodeContests-O](https://huggingface.co/datasets/caijanfeng/CodeContests-O) by Cai et al. ([arXiv:2601.13682](https://arxiv.org/abs/2601.13682)), revision `1a765191567b429f633bbd1c6e67b5890dfaf267`. The dataset card specifies [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This subset was filtered, split, and reduced to at most eight cases per problem. Recorded prompts incorporate those statements.
