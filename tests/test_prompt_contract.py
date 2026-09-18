"""The answer-tag contract (instruction, format check, recorded identity) and answer extraction."""
from __future__ import annotations

import hashlib
import json

import pytest

from lwl.evaluation.prompt_contract import (
    ANSWER_FORMAT_CONTRACT,
    DEFAULT_PROMPT_CONTRACT,
    PROMPT_CONTRACT_SCHEMA_VERSION,
    PromptContract,
    format_question,
    load_prompt_contract_from_legacy_config_run_manifest,
    load_prompt_contract_from_run_manifest,
    prompt_contract_metadata,
    resolve_prompt_contract,
    response_satisfies_contract,
)
from lwl.rewards.answer import extract_answer_span


VALID_RESPONSES = [
    ("<answer>7</answer>", "one well-formed tag"),
    ("reasoning first.\n<answer>7</answer>", "text before the tag"),
    ("<ANSWER>7</ANSWER>", "tags in upper case"),
    ("<answer>\n  7\n</answer>", "whitespace inside the tag"),
    ("<answer>7</answer>\n\n", "trailing whitespace after the tag"),
]

INVALID_RESPONSES = [
    ("7", "no tag at all"),
    ("<answer>7", "no closing tag"),
    ("<answer>7</answer", "a truncated closing tag"),
    ("answer>7</answer>", "a truncated opening tag"),
    ("<answer>7</answer><answer>8</answer>", "two tags"),
    ("<answer>7 <answer>8</answer>", "a second opening tag inside the span"),
    ("<answer></answer>", "an empty tagged span"),
    ("<answer>   </answer>", "a whitespace-only tagged span"),
    ("<answer>7</answer> and more reasoning", "text after the closing tag"),
    ("", "an empty response"),
]


@pytest.mark.parametrize("response, description", VALID_RESPONSES)
def test_responses_that_satisfy_the_contract(response, description):
    assert response_satisfies_contract(response), (
        f"{description}: {response!r} should satisfy the contract"
    )


@pytest.mark.parametrize("response, description", INVALID_RESPONSES)
def test_responses_that_break_the_contract(response, description):
    assert not response_satisfies_contract(response), (
        f"{description}: {response!r} should not satisfy the contract"
    )


@pytest.mark.parametrize("response, description", VALID_RESPONSES)
def test_a_response_that_satisfies_the_contract_is_read_from_its_tag(response, description):
    extracted = extract_answer_span(response)
    assert (extracted.span, extracted.extraction_level) == ("7", "tag"), (
        f"{description}: {response!r} extracted {extracted.span!r} at level "
        f"{extracted.extraction_level!r}, expected '7' from the tag"
    )
    assert extracted.extractor_valid and not extracted.extraction_fallback_used, (
        f"{description}: a tagged answer should be a clean extraction"
    )


# (response, span, level, clean): clean means the extractor found a marked answer.
EXTRACTION_CASES = [
    (r"reasoning \boxed{7} and then <answer>9</answer>", "9", "tag", True, "a tag wins over a box"),
    ("<answer>3</answer> then <answer>9</answer>", "9", "tag", True, "the last of two tags"),
    (r"steps \boxed{\frac{1}{2}}", r"\frac{1}{2}", "boxed", True, "a box with nested braces"),
    (r"\boxed{3} then \boxed{9}", "9", "boxed", True, "the last of two boxes"),
    ("some work\nFinal answer: 7", "7", "line", True, "a final-answer line"),
    ("Answer: 4\nmore text", "4", "line", True, "an answer line before trailing text"),
    ("some work\n7", "7", "lastline", False, "an unmarked last line"),
    ("<answer>3</answer", "<answer>3</answer", "lastline", False, "a truncated closing tag"),
    ("", "", "fulltext", False, "an empty response"),
]


@pytest.mark.parametrize("response, span, level, clean, description", EXTRACTION_CASES)
def test_answer_extraction_order(response, span, level, clean, description):
    extracted = extract_answer_span(response)
    assert (extracted.span, extracted.extraction_level) == (span, level), (
        f"{description}: {response!r} extracted {extracted.span!r} at level "
        f"{extracted.extraction_level!r}, expected {span!r} at level {level!r}"
    )
    assert extracted.extractor_valid is clean, (
        f"{description}: extractor_valid was {extracted.extractor_valid}, expected {clean}"
    )
    assert extracted.extraction_fallback_used is not clean, (
        f"{description}: the fallback flag should be {not clean}"
    )


def test_the_question_carries_the_instruction():
    formatted = format_question("  What is the x-coordinate?  ")
    assert formatted == f"What is the x-coordinate?\n\n{ANSWER_FORMAT_CONTRACT}", (
        f"the question should be stripped and followed by the instruction, got {formatted!r}"
    )
    assert "<answer>" in ANSWER_FORMAT_CONTRACT and "</answer>" in ANSWER_FORMAT_CONTRACT, (
        "the instruction should name both answer tags"
    )


def test_the_instruction_wording_is_pinned():
    # Every run was trained and scored under this exact wording; any edit changes the prompt.
    assert ANSWER_FORMAT_CONTRACT == (
        "Give your final answer inside <answer> and </answer> tags. "
        "Keep the tagged span to the shortest answer that resolves the question."
    ), f"the answer-format instruction changed: {ANSWER_FORMAT_CONTRACT!r}"
    assert DEFAULT_PROMPT_CONTRACT.instruction == ANSWER_FORMAT_CONTRACT, (
        "the default contract should carry the pinned instruction"
    )
    assert DEFAULT_PROMPT_CONTRACT.response_format == "single_final_answer_tag", (
        f"unexpected response format {DEFAULT_PROMPT_CONTRACT.response_format!r}"
    )


RECORDED_CONTRACT_SHA256 = "7ac39f53a2a824490fc5ee22671a888d2d79d55e1d8351919006d7d71c7a8f3f"


def test_the_default_contract_hash_is_the_one_every_evaluation_recorded():
    # Any change to a contract field, the schema string included, breaks the match with the records.
    assert DEFAULT_PROMPT_CONTRACT.sha256 == RECORDED_CONTRACT_SHA256, (
        f"the default contract hashes to {DEFAULT_PROMPT_CONTRACT.sha256}, "
        f"the evaluations recorded {RECORDED_CONTRACT_SHA256}"
    )
    assert prompt_contract_metadata()["prompt_contract_sha256"] == RECORDED_CONTRACT_SHA256, (
        "the metadata written with each evaluation should carry the recorded hash"
    )


def test_the_default_contract_is_the_answer_tag_contract():
    assert DEFAULT_PROMPT_CONTRACT.contract_id == "answer-tags-v1", (
        f"unexpected contract id {DEFAULT_PROMPT_CONTRACT.contract_id!r}"
    )
    assert resolve_prompt_contract(None) == DEFAULT_PROMPT_CONTRACT, (
        "no contract given should resolve to the default one"
    )
    assert resolve_prompt_contract(DEFAULT_PROMPT_CONTRACT.to_dict()) == DEFAULT_PROMPT_CONTRACT, (
        "the default contract should survive a round trip through its dict form"
    )


def test_the_recorded_identity_follows_the_wording():
    metadata = prompt_contract_metadata()
    assert metadata["prompt_contract_id"] == "answer-tags-v1", (
        f"unexpected contract id {metadata['prompt_contract_id']!r}"
    )
    canonical = json.dumps(DEFAULT_PROMPT_CONTRACT.to_dict(), sort_keys=True, separators=(",", ":"))
    assert metadata["prompt_contract_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest(), (
        "the recorded hash should be the sha256 of the contract's canonical JSON"
    )
    reworded = PromptContract(
        contract_id="answer-tags-v1",
        instruction=ANSWER_FORMAT_CONTRACT + " Be brief.",
        response_format="single_final_answer_tag",
    )
    assert reworded.sha256 != DEFAULT_PROMPT_CONTRACT.sha256, (
        "a different instruction must produce a different hash"
    )


def test_a_contract_with_the_wrong_fields_is_rejected():
    fields = DEFAULT_PROMPT_CONTRACT.to_dict()
    with pytest.raises(ValueError, match="missing"):
        resolve_prompt_contract({key: value for key, value in fields.items() if key != "instruction"})
    with pytest.raises(ValueError, match="extra"):
        resolve_prompt_contract({**fields, "extra": "x"})
    with pytest.raises(ValueError, match="schema"):
        resolve_prompt_contract({**fields, "schema_version": "something-else.v9"})


def test_an_unsupported_response_format_is_rejected():
    contract = PromptContract(
        contract_id="answer-tags-v1",
        instruction=ANSWER_FORMAT_CONTRACT,
        response_format="two_tags",
    )
    with pytest.raises(ValueError, match="response format"):
        response_satisfies_contract("<answer>7</answer>", contract)


def test_a_run_manifest_must_carry_the_hash_of_its_own_contract(tmp_path):
    contract = DEFAULT_PROMPT_CONTRACT
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"prompt_contract": contract.to_dict(), "prompt_contract_sha256": contract.sha256}),
        encoding="utf-8",
    )
    assert load_prompt_contract_from_run_manifest(path) == contract, (
        "a run manifest carrying the contract and its hash should load"
    )
    path.write_text(
        json.dumps({"prompt_contract": contract.to_dict(), "prompt_contract_sha256": "0" * 64}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prompt_contract_from_run_manifest(path)
    path.write_text(json.dumps({"schema_version": PROMPT_CONTRACT_SCHEMA_VERSION}), encoding="utf-8")
    with pytest.raises(ValueError, match="no prompt_contract"):
        load_prompt_contract_from_run_manifest(path)


def test_an_older_run_recovers_its_contract_from_its_hashed_config(tmp_path):
    instruction = "Reason step by step, then give the answer inside <answer> and </answer> tags."
    config = json.dumps({"model": {"policy": {"system_prompt": instruction}}}).encode("utf-8")
    (tmp_path / "config.json").write_bytes(config)
    manifest = tmp_path / "run.json"
    manifest.write_text(
        json.dumps({"config_path": "config.json", "config_hash": hashlib.sha256(config).hexdigest()}),
        encoding="utf-8",
    )
    contract = load_prompt_contract_from_legacy_config_run_manifest(manifest, tmp_path)
    assert contract.instruction == instruction, (
        f"the instruction should come from the config, got {contract.instruction!r}"
    )
    (tmp_path / "config.json").write_bytes(config + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_prompt_contract_from_legacy_config_run_manifest(manifest, tmp_path)
