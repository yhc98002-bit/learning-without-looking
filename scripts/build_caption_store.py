#!/usr/bin/env python3
"""Generate the question-blind caption store for one or more image directories with vLLM.

Images are deduplicated by sha256 and captioned greedily with the fixed prompt; the output is
written incrementally so an interrupted run can be resumed from its own prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lwl.captions.store import (
    CAPTION_DECODING,
    CAPTION_PROMPT,
    CAPTION_PROMPT_SHA256,
    SCHEMA_VERSION,
    discover_images,
    load_validated_caption_prefix,
)


CAPTION_MAX_NEW_TOKENS = 384
SAMPLING_SEED = 0


def discover_image_roots(input_dirs: list[Path]) -> list[dict[str, Any]]:
    """Merge several image roots into one list of entries, one per distinct image hash."""
    if not input_dirs:
        raise ValueError("at least one image root is required")
    paths_by_hash: dict[str, set[str]] = {}
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"image root is absent: {input_dir}")
        for item in discover_images(input_dir):
            paths = [str(item["image_path"]), *map(str, item["duplicate_paths"])]
            paths_by_hash.setdefault(str(item["image_sha256"]), set()).update(paths)
    if not paths_by_hash:
        raise ValueError("image roots contain no readable images")
    rows = []
    for digest, paths in sorted(paths_by_hash.items()):
        ordered = sorted(paths)
        rows.append(
            {
                "image_sha256": digest,
                "image_path": ordered[0],
                "duplicate_paths": ordered[1:],
            }
        )
    return rows


def build_request(processor: Any, image_path: str) -> tuple[dict[str, Any], Image.Image]:
    """Build the chat request for one image; the opened image is returned so it can be closed."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    return {"prompt": prompt, "multi_modal_data": {"image": image}}, image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--caption-model-id", required=True)
    parser.add_argument("--caption-model-revision", required=True)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(2, 4), default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=CAPTION_MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite caption-store output: {args.output}")
    if args.max_new_tokens != CAPTION_MAX_NEW_TOKENS:
        raise ValueError("caption token budget must remain 384")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"caption model directory is absent: {args.model_path}")

    items = discover_image_roots(args.input_dir)
    resume_lines = (
        load_validated_caption_prefix(
            args.resume_from,
            items,
            model_path=args.caption_model_id,
            max_new_tokens=args.max_new_tokens,
        )
        if args.resume_from
        else []
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for line in resume_lines:
            handle.write(line + "\n")
        handle.flush()
    if len(resume_lines) == len(items):
        print(json.dumps({"n_images": len(items), "resumed": len(resume_lines)}))
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    llm = LLM(
        model=str(args.model_path),
        tokenizer=str(args.model_path),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        n=1,
        max_tokens=args.max_new_tokens,
        seed=SAMPLING_SEED,
    )
    input_hash = hashlib.sha256(
        "\n".join(str(path) for path in args.input_dir).encode("utf-8")
    ).hexdigest()

    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(len(resume_lines), len(items), args.batch_size):
            batch = items[start : start + args.batch_size]
            requests: list[dict[str, Any]] = []
            opened_images: list[Image.Image] = []
            for item in batch:
                request, image = build_request(processor, str(item["image_path"]))
                requests.append(request)
                opened_images.append(image)
            try:
                outputs = llm.generate(requests, sampling, use_tqdm=False)
            finally:
                for image in opened_images:
                    image.close()
            for item, output in zip(batch, outputs):
                caption = output.outputs[0].text.strip()
                if not caption:
                    raise ValueError(f"empty caption for {item['image_sha256']}")
                row = {
                    "schema_version": SCHEMA_VERSION,
                    **item,
                    "caption": caption,
                    "caption_model_path": args.caption_model_id,
                    "caption_model_revision": args.caption_model_revision,
                    "caption_prompt": CAPTION_PROMPT,
                    "caption_prompt_sha256": CAPTION_PROMPT_SHA256,
                    "max_new_tokens": args.max_new_tokens,
                    "decoding": CAPTION_DECODING,
                    "source_roots_sha256": input_hash,
                    "tensor_parallel_width": args.tensor_parallel_size,
                }
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
                handle.flush()
            print(
                json.dumps(
                    {
                        "processed": min(start + len(batch), len(items)),
                        "resumed": len(resume_lines),
                        "total": len(items),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
