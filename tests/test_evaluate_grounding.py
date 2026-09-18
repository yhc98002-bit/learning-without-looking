"""The grounding evaluation driver end to end on two released-format pairs, with a stub model
standing in for the checkpoint: sharding, scoring, resuming and aggregation."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from lwl.evaluation.prompt_contract import ANSWER_FORMAT_CONTRACT
from lwl.paths import repo_root

DRIVER = repo_root() / "scripts" / "evaluate_grounding.sh"
AGGREGATE = repo_root() / "scripts" / "aggregate_evaluation.py"

# torch, transformers and qwen_vl_utils as far as scripts/evaluate.py uses them. The model
# answers from a table keyed by the image file name and logs every prompt it is given.
STUBS = {
    "torch.py": '''
import contextlib

bfloat16 = "bfloat16"


def inference_mode():
    return contextlib.nullcontext()
''',
    "qwen_vl_utils.py": '''
def process_vision_info(messages):
    images = [part["image"] for message in messages for part in message["content"] if part["type"] == "image"]
    return images, None
''',
    "transformers.py": '''
import json
import os
from pathlib import Path

import numpy as np

RESPONSES = json.loads(Path(os.environ["STUB_RESPONSES"]).read_text())
DECODED = []


def set_seed(seed):
    pass


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"][-1]["text"]

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        return Inputs(input_ids=np.zeros((1, 4), dtype=np.int64), images=images or [], prompt=text[0])

    def batch_decode(self, out, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return [DECODED[int(token)] for token in out[:, 0]]


class Model:
    device = "cpu"

    def generate(self, input_ids, images, prompt, do_sample, max_new_tokens):
        name = Path(str(images[0])).name if images else ""
        DECODED.append(RESPONSES.get(name, "<answer>none</answer>"))
        with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as log:
            log.write(json.dumps({"image": str(images[0]) if images else None, "prompt": prompt}) + "\\n")
        return np.concatenate([input_ids, [[len(DECODED) - 1]]], axis=1)


class AutoProcessor:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return Processor()


class Qwen2_5_VLForConditionalGeneration:
    @staticmethod
    def from_pretrained(path, **kwargs):
        return Model()
''',
}

# Two pairs in the released manifest form: the model gets the first right and collapses the second.
PAIRS = [
    {
        "pair_id": "pair-coordinate", "template_id": "coordinate_register_twenty_point_x_v02",
        "category": "geometry_coordinate_indexing", "question": "What is the x-coordinate of point Y6?",
        "answer_a": "7", "answer_b": "-1",
    },
    {
        "pair_id": "pair-table", "template_id": "header_cued_table_code_v02",
        "category": "document_header_indexing",
        "question": "What is the 2-character code at row Case-591 and column G4?",
        "answer_a": "MX", "answer_b": "TX",
    },
]
RESPONSES = {
    "pair-coordinate_a.png": "The point is at x = 7.\n<answer>7</answer>",
    "pair-coordinate_b.png": "<answer>-1</answer>",
    "pair-table_a.png": "<answer>MX</answer>",
    "pair-table_b.png": "<answer>MX</answer>",
}


@pytest.fixture
def released_suite(tmp_path):
    """A data directory holding a two-pair suite, and an environment that runs the stub model."""
    data = tmp_path / "data"
    suite = data / "grounding" / "suite"
    rows = []
    for index, pair in enumerate(PAIRS):
        row = dict(pair)
        for side, shade in (("a", 40), ("b", 200)):
            image = suite / "images" / f"{pair['pair_id']}_{side}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((24, 32, 3), shade + index, dtype=np.uint8), mode="RGB").save(image)
            row[f"image_{side}_path"] = f"data/grounding/suite/images/{image.name}"
            row[f"changed_region_mask_{side}"] = f"data/grounding/suite/masks/{pair['pair_id']}.png"
        rows.append(row)
    (suite / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, source in STUBS.items():
        (stubs / name).write_text(source.lstrip(), encoding="utf-8")
    # The driver calls `python`; point it at this interpreter (a symlink would leave a venv).
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    (shim / "python").chmod(0o755)
    (tmp_path / "responses.json").write_text(json.dumps(RESPONSES), encoding="utf-8")

    env = dict(os.environ)
    env.update(
        LWL_DATA=str(data),
        PATH=f"{shim}{os.pathsep}{env.get('PATH', '')}",
        PYTHONPATH=os.pathsep.join([str(stubs), str(repo_root())]),
        STUB_RESPONSES=str(tmp_path / "responses.json"),
        STUB_LOG=str(tmp_path / "prompts.jsonl"),
    )
    return tmp_path, env


def run_driver(env, out, *extra):
    return subprocess.run(
        ["bash", str(DRIVER), "--model", "stub-model", "--out", str(out), "--instruments", "suite",
         "--gpus", "0 1", *extra],
        cwd=out.parent, env=env, capture_output=True, text=True, timeout=300,
    )


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


pytestmark = pytest.mark.skipif(shutil.which("bash") is None or not DRIVER.exists(),
                                reason="needs bash and the scripts directory")


def test_the_driver_scores_both_pairs_across_two_shards(released_suite):
    root, env = released_suite
    out = root / "eval"
    result = run_driver(env, out)
    assert result.returncode == 0, f"the driver failed:\n{result.stdout}\n{result.stderr}"

    shards = [read_jsonl(out / "suite" / "shards" / f"shard_{index}.jsonl") for index in (0, 1)]
    assert [[row["pair_id"] for row in shard] for shard in shards] == [["pair-coordinate"], ["pair-table"]], (
        "each of the two shards should hold one pair, in manifest order"
    )
    first, second = shards[0][0], shards[1][0]
    assert first["pair_correct"] and first["correct_a"] and first["correct_b"], (
        f"both members of the first pair were answered right: {first['answer_a']}, {first['answer_b']}"
    )
    assert (second["correct_a"], second["correct_b"], second["pair_correct"], second["collapsed"]) == (
        True, False, False, True,
    ), "the second pair gave one answer to both members and should score as collapsed"
    assert first["eval_image_a_path"].startswith(env["LWL_DATA"]), (
        f"the images should be read from LWL_DATA, got {first['eval_image_a_path']}"
    )

    for index in (0, 1):
        metrics = json.loads((out / "suite" / "metrics" / f"shard_{index}.json").read_text())
        assert metrics["num_shards"] == 2.0 and metrics["n_pairs"] == 1.0, f"shard {index}: {metrics}"
    prompts = read_jsonl(root / "prompts.jsonl")
    assert len(prompts) == 4, f"four members should reach the model, found {len(prompts)}"
    assert all(entry["prompt"].endswith(ANSWER_FORMAT_CONTRACT) for entry in prompts), (
        "every question should carry the answer-format instruction"
    )

    again = run_driver(env, out)
    assert again.returncode == 0, f"the resumed driver failed:\n{again.stdout}\n{again.stderr}"
    assert again.stdout.count("complete, skipped") == 2, f"both shards should be skipped:\n{again.stdout}"
    assert len(read_jsonl(root / "prompts.jsonl")) == 4, "a resumed run should not query the model again"

    aggregate = subprocess.run(
        [sys.executable, str(AGGREGATE), "--inputs", str(out / "suite" / "shards" / "shard_*.jsonl"),
         "--output", str(out / "suite" / "metrics.json"), "--bootstrap", "50", "--permutations", "50"],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert aggregate.returncode == 0, f"aggregation failed: {aggregate.stderr}"
    combined = json.loads((out / "suite" / "metrics.json").read_text())
    assert (combined["n_pairs"], combined["pair_accuracy"], combined["member_accuracy"]) == (2.0, 0.5, 0.75), (
        f"two pairs, one right and one collapsed: {combined['n_pairs']}, {combined['pair_accuracy']}, "
        f"{combined['member_accuracy']}"
    )


def test_the_gray_condition_replaces_both_images(released_suite):
    root, env = released_suite
    out = root / "eval_gray"
    result = run_driver(env, out, "--condition", "gray")
    assert result.returncode == 0, f"the driver failed:\n{result.stdout}\n{result.stderr}"
    rows = read_jsonl(out / "suite" / "shards" / "shard_0.jsonl") + read_jsonl(out / "suite" / "shards" / "shard_1.jsonl")
    cache = out / "suite" / "gray_image_cache"
    for row in rows:
        for side in ("a", "b"):
            rendered = row[f"eval_image_{side}_path"]
            assert rendered.startswith(str(cache)), f"{row['pair_id']} {side} was not rendered gray: {rendered}"
            with Image.open(rendered) as image:
                assert np.all(np.asarray(image.convert("RGB")) == 128), f"{rendered} is not a flat gray field"
    assert not any(row["pair_correct"] for row in rows), "the stub cannot answer from a gray field"
