# Same Reward, Different Skills

[![Hugging Face dataset](https://img.shields.io/badge/Hugging%20Face-dataset-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Despaireyes613/learning-without-looking)

Code and released data for a study of what reinforcement learning with verifiable rewards (RLVR)
teaches a vision-language model about looking.

RLVR raises vision-language benchmark scores even when visual information is removed during training:
with images at test, blind-trained models recover roughly half of the real-image gain at 3B and nearly
four fifths at 7B. Under prolonged training with real images, benchmark accuracy peaks and then stays
near-flat while grounding falls below the base model's.

Both follow from one fact: an image in the prompt is not an image in the learning signal. A training
problem is visually resolvable when correct answers require the image and the task remains learnable.
We build counterfactual coordinate scenes whose answers change with the image while the question stays
fixed and never names the target. With the reward unchanged, standard GRPO learns to find the target
and read it: discovery accuracy on held-out scenes rises from 0.425 to 0.875, drops to zero when the
test image is replaced by a gray canvas, is not recovered by a matched-budget run trained without visual
information, and transfers to independent grounding tasks.

## What is here

```
lwl/scenes/         the coordinate scene program: densities, roles, twins, the four cue levels
lwl/grounding/      the held-out counterfactual instruments and their regenerated twin
lwl/corpora/        Geometry3K and ViRL39K preparation, decontamination, the dose mixtures
lwl/captions/       the question-blind caption store used by the caption-condition arm
lwl/audit/          the visual-necessity audit: sampling, per-item summaries
lwl/rewards/        the training reward and its answer matcher
lwl/evaluation/     prompt contract, answer scoring, image conditions, benchmark adapters
lwl/analysis/       the statistics and the per-table rebuilders
configs/train/      the training recipes, one file per run or training segment
patches/easyr1/     the modifications to the training framework, as diffs
scripts/            command-line entry points for every stage
tests/              CPU tests
```

## Install

Python 3.10 to 3.12 (numpy 1.26.4 has no wheels for 3.13):

```bash
git clone https://github.com/yhc98002-bit/learning-without-looking
cd learning-without-looking
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt    # analysis and tests
pip install -e .
pytest -q
```

Tests that need the released data are skipped until it is downloaded.

## Rebuild the paper's numbers

The evaluation outputs of every run the paper reports are published item by item, so every table and
figure can be rebuilt on a laptop:

```bash
python scripts/fetch_data.py --part predictions     # about 22 MB
python scripts/reproduce.py all --check
```

Each target writes a CSV and a JSON into `results/`, and `--check` compares every rebuilt value with
the value printed in the paper.

## Re-run the pipeline

Training and evaluation need GPUs and the full environment:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python scripts/fetch_data.py --part instruments --part corpora --extract
python scripts/fetch_virl39k.py                     # the ViRL39K images, from the upstream release
bash scripts/setup_easyr1.sh                        # clone and patch the trainer
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir artifacts/models/Qwen2.5-VL-7B-Instruct
```

The reported step-100 constructed checkpoints come from a 30-step config and three segment configs,
run in order. Only weights are saved, so each segment starts from the merged weights of the one
before it:

```bash
R=constructed-standard-7b-run1
bash scripts/train.sh $R                            # steps 0-30
python scripts/merge_checkpoint.py --local_dir checkpoints/$R/global_step_30/actor
bash scripts/train.sh $R-steps30-50
python scripts/merge_checkpoint.py --local_dir checkpoints/$R-steps30-50/global_step_20/actor
bash scripts/train.sh $R-steps50-75
python scripts/merge_checkpoint.py --local_dir checkpoints/$R-steps50-75/global_step_25/actor
bash scripts/train.sh $R-steps75-100
python scripts/merge_checkpoint.py --local_dir checkpoints/$R-steps75-100/global_step_25/actor
```

Each checkpoint is then evaluated on the scenes (every cue level, real and gray images) and on the
grounding suite and its twin:

```bash
MODEL=checkpoints/$R-steps75-100/global_step_25/actor/huggingface     # step 100
bash scripts/evaluate_scenes.sh --model $MODEL --out outputs/scenes --split confirmatory \
    --levels "l3 l2 l1 probe" --conditions "real gray"
bash scripts/evaluate_grounding.sh --model $MODEL --out outputs/grounding \
    --instruments "suite twin" --gpus "0 1 2 3"
python scripts/aggregate_evaluation.py --inputs "outputs/grounding/suite/shards/*.jsonl" \
    --output outputs/grounding/suite/metrics.json
```

The long-horizon runs are also trained in segments, but save full checkpoints, so each segment
resumes the previous one exactly; the headers of `configs/train/long-horizon-*.yaml` give the order.

The 3B recipes start from `artifacts/models/Qwen2.5-VL-3B-Instruct`. To evaluate the released
checkpoint instead, fetch it with `python scripts/fetch_data.py --part checkpoint` and pass
`--model checkpoints/constructed-standard-7b-run1-step100`.

The ViRL39K images are not redistributed. `scripts/fetch_virl39k.py` takes them from the pinned
upstream release and checks each against the SHA-256 recorded in the training rows.

`scripts/build_scenes.py` regenerates the scene sets from the scene program,
`scripts/build_training_corpus.py` builds the constructed training rows from the training scenes,
and `scripts/build_mixtures.py` mixes them with the filtered ViRL39K rows; the published corpora let
you skip these steps. The grounding suite and its twin are released frozen; their generators are
`lwl/grounding/build_suite.py` and `lwl/grounding/build_twin.py`.

## Released data

`scripts/fetch_data.py` downloads from the dataset repository (badge at the top):

| part | size | contents |
|---|---|---|
| `predictions` | 22 MB | per-item evaluation outputs, plus the training streams and audit rows the rebuilders read |
| `instruments` | 610 MB | the scene sets, the grounding suite and its twin, the audit inputs |
| `corpora` | 150 MB | the filtered upstream corpora, the dose mixtures, the caption stores |
| `checkpoint` | 16 GB | the 7B model trained on the constructed corpus with the standard reward |

## Notes on the numbers

- Pair files carry the repository's current scorer in `correct_a`, `correct_b` and `pair_correct`,
  and the flags recorded at evaluation time in the `*_as_logged` fields. Item files carry `correct`
  (the canonical matcher) and `correct_reward_matcher` (the training reward's matcher). Audit files
  carry counts and rates under both matchers. The tables use the current pair scorer and the
  canonical item matcher, with two exceptions: the degraded-set overlap (Figure 2c, Table E.2) uses
  the `*_as_logged` flags, and the resolvability audit of the training corpora (Section 5,
  Figure 3a and the mixture masses) uses the reward matcher (`*_reward_matcher`).
- Six printed values cannot be rebuilt from the released files, such as the training-reward curve,
  which comes from the trainer's own log. `--check` reports them as not rebuildable.

## Licence

Apache-2.0 for the code; see `LICENSE` and the third-party terms in `NOTICE`.
