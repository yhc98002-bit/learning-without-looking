"""The answer-tag contract: the instruction appended to every question, and the check
that a response carries exactly one well-formed answer span.

Each run records the contract it used, so evaluation can reproduce the prompt wording
and the format check the run was trained and scored under.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


ANSWER_FORMAT_CONTRACT = (
    "Give your final answer inside <answer> and </answer> tags. "
    "Keep the tagged span to the shortest answer that resolves the question."
)
# Part of the hashed contract: every evaluation recorded the hash computed with this exact string.
PROMPT_CONTRACT_SCHEMA_VERSION = "blind-gains.prompt-contract.v1"


@dataclass(frozen=True)
class PromptContract:
    contract_id: str
    instruction: str
    response_format: str
    schema_version: str = PROMPT_CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_PROMPT_CONTRACT = PromptContract(
    contract_id="answer-tags-v1",
    instruction=ANSWER_FORMAT_CONTRACT,
    response_format="single_final_answer_tag",
)

PromptContractLike = PromptContract | Mapping[str, Any] | None


def resolve_prompt_contract(value: PromptContractLike = None) -> PromptContract:
    if value is None:
        return DEFAULT_PROMPT_CONTRACT
    if isinstance(value, PromptContract):
        return value
    required = {"contract_id", "instruction", "response_format", "schema_version"}
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        raise ValueError(f"prompt contract fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    contract = PromptContract(**{key: str(value[key]) for key in required})
    if contract.schema_version != PROMPT_CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported prompt contract schema: {contract.schema_version}")
    return contract


def prompt_contract_metadata(value: PromptContractLike = None) -> dict[str, str]:
    contract = resolve_prompt_contract(value)
    return {
        "prompt_contract_id": contract.contract_id,
        "prompt_contract_sha256": contract.sha256,
    }


def response_satisfies_contract(response: Any, value: PromptContractLike = None) -> bool:
    contract = resolve_prompt_contract(value)
    text = str(response).strip()
    if contract.response_format != "single_final_answer_tag":
        raise ValueError(f"unsupported response format: {contract.response_format}")
    if len(re.findall(r"<answer\b", text, re.IGNORECASE)) != 1:
        return False
    if len(re.findall(r"</answer\s*>", text, re.IGNORECASE)) != 1:
        return False
    match = re.fullmatch(r"(?s).*<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE)
    return bool(match and match.group(1).strip())


def load_prompt_contract_from_run_manifest(path: str | Path) -> PromptContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    contract = payload.get("prompt_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"run manifest has no prompt_contract mapping: {path}")
    resolved = resolve_prompt_contract(contract)
    recorded_hash = payload.get("prompt_contract_sha256")
    if recorded_hash != resolved.sha256:
        raise ValueError(
            f"run manifest prompt contract hash mismatch: expected {resolved.sha256}, found {recorded_hash}"
        )
    return resolved


def load_prompt_contract_from_legacy_config_run_manifest(
    path: str | Path, repo_root: str | Path
) -> PromptContract:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("prompt_contract"), dict):
        raise ValueError(
            f"legacy config resolution is not allowed for a run with an embedded contract: {path}"
        )
    config_value = payload.get("config_path")
    expected_hash = payload.get("config_hash")
    if not isinstance(config_value, str) or not config_value:
        raise ValueError(f"legacy run manifest has no config_path: {path}")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError(f"legacy run manifest has no valid config_hash: {path}")
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = Path(repo_root) / config_path
    config_bytes = config_path.read_bytes()
    observed_hash = hashlib.sha256(config_bytes).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError(
            f"legacy run config hash mismatch: expected {expected_hash}, found {observed_hash}"
        )
    config = json.loads(config_bytes)
    models = config.get("model")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"legacy run config has no model mapping: {config_path}")
    prompts = {
        str(model.get("system_prompt", "")).strip()
        for model in models.values()
        if isinstance(model, dict) and str(model.get("system_prompt", "")).strip()
    }
    if len(prompts) != 1:
        raise ValueError(
            f"legacy run config must contain exactly one nonempty system_prompt: {config_path}"
        )
    instruction = prompts.pop()
    if "<answer>" not in instruction.lower() or "</answer>" not in instruction.lower():
        raise ValueError(
            f"legacy system_prompt does not state the answer-tag contract: {config_path}"
        )
    return PromptContract(
        contract_id="answer-tags-v1",
        instruction=instruction,
        response_format="single_final_answer_tag",
    )


def format_question(question: str, contract: PromptContractLike = None) -> str:
    resolved = resolve_prompt_contract(contract)
    return f"{question.strip()}\n\n{resolved.instruction}"
