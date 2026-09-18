"""Test-time image conditions: the real image, a flat gray field, per-item noise, the removed image
and the question-blind caption."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lwl.evaluation.conditioned_inputs import build_conditioned_messages
from lwl.evaluation.image_conditions import IMAGE_MODES, materialize_image


FORMAT_PROMPT = "{{ content }}"
QUESTION = "What is the x-coordinate of the highest point?"

# A 16x12 PNG kept as bytes, so its digest, and with it every rendered file name, is fixed.
FIXED_SOURCE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAMCAIAAADkharWAAAAIklEQVR4nGM88Z+BgZEBCohgMzGQCIaDBkaG/1GjoUQIAAB2ZQk5iPqCMgAAAABJRU5ErkJggg=="
)
FIXED_SEED = 5
# (rendered file name, sha256 of the rendered RGB pixels) for the fixed source and seed.
PINNED_RENDERS = {
    "gray": ("6c7ec3b8618c820b.png", "af1857bf5516aa3e2e39b6842559746fa7b45daa8dc4cc6675ad86e0cfe425b9"),
    "noise": ("ba6afdd8840e4fb1.png", "5858236eff7c1d25c7878c990ab1e70e0d4dc0f5ff37c443e4e46efd9135a9d9"),
}
# sha256 of the removed-image messages for the fixed row, as canonical JSON.
PINNED_REMOVED_MESSAGES = "6fe9d4855751e781c08e08dfd51fbb21738dc712cada59a5c568955f0957aca6"


@pytest.fixture
def source_image(tmp_path):
    """A small picture with enough structure to tell it apart from a flat field."""
    pixels = np.zeros((12, 16, 3), dtype=np.uint8)
    pixels[:6, :, 0] = 200
    pixels[6:, :, 2] = 90
    pixels[:, ::4, 1] = 255
    path = tmp_path / "source.png"
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
    return path


def pixels_of(path):
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"))


def row_with_image(source_image):
    return {
        "row_index": 3,
        "problem": f"<image>\n{QUESTION}",
        "images": [{"path": str(source_image), "sha256": "a" * 64}],
    }


def rendered_view(messages):
    """The message content with every image replaced by its pixels, so two cache directories compare."""
    view = []
    for part in messages[0]["content"]:
        if part["type"] == "image":
            view.append(("image", pixels_of(part["image"]).tobytes()))
        else:
            view.append(("text", part["text"]))
    return view


def test_the_real_condition_passes_the_image_through(source_image, tmp_path):
    returned = materialize_image(str(source_image), "real", tmp_path / "cache")
    assert returned == str(source_image), f"the real condition should return the input path, got {returned!r}"
    assert not (tmp_path / "cache").exists(), "the real condition should render nothing"


@pytest.mark.parametrize("condition", ["gray", "none", "noise"])
def test_a_condition_is_deterministic_and_differs_from_the_real_input(source_image, tmp_path, condition):
    row = row_with_image(source_image)
    first, _ = build_conditioned_messages(row, FORMAT_PROMPT, condition, tmp_path / "one", noise_seed=5)
    again, _ = build_conditioned_messages(row, FORMAT_PROMPT, condition, tmp_path / "two", noise_seed=5)
    real, _ = build_conditioned_messages(row, FORMAT_PROMPT, "real", tmp_path / "real", noise_seed=5)
    assert rendered_view(first) == rendered_view(again), (
        f"two {condition} inputs built with the same seed differ"
    )
    assert rendered_view(first) != rendered_view(real), f"the {condition} input equals the real one"
    texts = [part["text"] for part in first[0]["content"] if part["type"] == "text"]
    assert any(QUESTION in text for text in texts), f"the question was lost under {condition}: {texts}"


def test_gray_is_a_flat_field_of_the_same_size(source_image, tmp_path):
    rendered = pixels_of(materialize_image(str(source_image), "gray", tmp_path / "cache"))
    assert rendered.shape == pixels_of(source_image).shape, (
        f"gray should keep the size of the image, got {rendered.shape}"
    )
    assert np.all(rendered == 128), f"gray should be 128 everywhere, found {np.unique(rendered)[:5]}"


def test_the_removed_condition_leaves_one_text_part_and_no_image(source_image, tmp_path):
    messages, paths = build_conditioned_messages(
        row_with_image(source_image), FORMAT_PROMPT, "none", tmp_path / "cache"
    )
    content = messages[0]["content"]
    assert [part["type"] for part in content] == ["text"], (
        f"the removed condition should leave one text part, got {[part['type'] for part in content]}"
    )
    assert content[0]["text"].strip() == QUESTION, (
        f"the removed condition should leave the question alone, got {content[0]['text']!r}"
    )
    assert paths == [], f"no image should have been rendered, got {paths}"


def test_noise_is_fixed_by_its_seed(source_image, tmp_path):
    first = pixels_of(materialize_image(str(source_image), "noise", tmp_path / "a", noise_seed=1))
    again = pixels_of(materialize_image(str(source_image), "noise", tmp_path / "b", noise_seed=1))
    other = pixels_of(materialize_image(str(source_image), "noise", tmp_path / "c", noise_seed=2))
    assert np.array_equal(first, again), "the same seed should give the same noise"
    assert not np.array_equal(first, other), "a different seed should give different noise"


@pytest.mark.parametrize("mode", ["mild", "medium", "severe"])
def test_a_degradation_keeps_the_size_and_changes_the_pixels(source_image, tmp_path, mode):
    rendered = pixels_of(materialize_image(str(source_image), mode, tmp_path / "cache"))
    original = pixels_of(source_image)
    assert rendered.shape == original.shape, f"{mode} changed the size to {rendered.shape}"
    assert not np.array_equal(rendered, original), f"{mode} left the image unchanged"


def test_the_rendered_file_is_reused(source_image, tmp_path):
    cache = tmp_path / "cache"
    first = materialize_image(str(source_image), "gray", cache)
    second = materialize_image(str(source_image), "gray", cache)
    assert first == second, f"the same request should reuse one file, got {first} and {second}"
    files = list(cache.glob("*.png"))
    assert len(files) == 1, f"one rendering should leave one file, found {len(files)}"
    assert Path(first).parent == cache, f"the rendering should sit in the cache, not at {first}"


def test_an_unknown_condition_is_rejected(source_image, tmp_path):
    assert "blurred" not in IMAGE_MODES, "the test needs a mode the module does not define"
    with pytest.raises(ValueError, match="unsupported image mode"):
        materialize_image(str(source_image), "blurred", tmp_path / "cache")
    with pytest.raises(ValueError, match="unsupported image condition"):
        build_conditioned_messages(row_with_image(source_image), FORMAT_PROMPT, "blurred", tmp_path / "cache")


def test_a_row_whose_markers_and_images_disagree_is_rejected(source_image, tmp_path):
    row = row_with_image(source_image)
    row["problem"] = f"<image><image>\n{QUESTION}"
    with pytest.raises(ValueError, match="image markers"):
        build_conditioned_messages(row, FORMAT_PROMPT, "real", tmp_path / "cache")


@pytest.fixture
def fixed_source(tmp_path):
    path = tmp_path / "fixed.png"
    path.write_bytes(FIXED_SOURCE_PNG)
    return path


@pytest.mark.parametrize("mode", sorted(PINNED_RENDERS))
def test_a_render_is_pinned_for_a_fixed_source_and_seed(fixed_source, tmp_path, mode):
    rendered = Path(materialize_image(str(fixed_source), mode, tmp_path / "cache", noise_seed=FIXED_SEED))
    name, digest = PINNED_RENDERS[mode]
    assert rendered.name == name, f"the {mode} render is named {rendered.name}, pinned {name}"
    found = hashlib.sha256(pixels_of(rendered).tobytes()).hexdigest()
    assert found == digest, f"the {mode} render has pixel digest {found}, pinned {digest}"


def test_the_removed_condition_is_pinned_for_a_fixed_row(fixed_source, tmp_path):
    messages, paths = build_conditioned_messages(
        row_with_image(fixed_source), FORMAT_PROMPT, "none", tmp_path / "cache", noise_seed=FIXED_SEED
    )
    found = hashlib.sha256(json.dumps(messages, sort_keys=True).encode("utf-8")).hexdigest()
    assert found == PINNED_REMOVED_MESSAGES, (
        f"the removed-image messages changed: {json.dumps(messages)} has digest {found}"
    )
    assert paths == [], f"no image should have been rendered, got {paths}"


def two_image_row(tmp_path):
    return {
        "row_index": 7,
        "problem": f"<image> <image>\n{QUESTION}",
        "images": [
            {"path": str(tmp_path / "first.png"), "sha256": "1" * 64},
            {"path": str(tmp_path / "second.png"), "sha256": "2" * 64},
        ],
    }


def test_the_caption_condition_puts_each_caption_in_place_of_its_image(tmp_path):
    captions = {"1" * 64: "Three labeled points on a grid.", "2" * 64: "A bar chart with four bars."}
    messages, paths = build_conditioned_messages(
        two_image_row(tmp_path), FORMAT_PROMPT, "caption", tmp_path / "cache", captions=captions
    )
    assert messages[0]["content"] == [
        {"type": "text", "text": "\n[Question-blind image description 1: Three labeled points on a grid.]\n"},
        {"type": "text", "text": " "},
        {"type": "text", "text": "\n[Question-blind image description 2: A bar chart with four bars.]\n"},
        {"type": "text", "text": f"\n{QUESTION}"},
    ], f"each image should become its caption, in order: {messages[0]['content']}"
    assert paths == [], f"the caption condition should render no image, got {paths}"
    assert not (tmp_path / "cache").exists(), "the caption condition should not touch the image cache"


def test_the_caption_condition_needs_a_caption_for_every_image(tmp_path):
    with pytest.raises(KeyError, match="missing fixed caption"):
        build_conditioned_messages(
            two_image_row(tmp_path), FORMAT_PROMPT, "caption", tmp_path / "cache",
            captions={"1" * 64: "Three labeled points on a grid."},
        )
    with pytest.raises(KeyError, match="missing fixed caption"):
        build_conditioned_messages(two_image_row(tmp_path), FORMAT_PROMPT, "caption", tmp_path / "cache")
