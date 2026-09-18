"""Adapters for the two non-Qwen families the paper evaluates, InternVL3 and Gemma-3:
prompt assembly, greedy generation, and the runtime facts each backend must report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from packaging.version import Version
from PIL import Image

from lwl.evaluation.prompt_contract import ANSWER_FORMAT_CONTRACT, format_question
from lwl.paths import resolve_data_reference


NONQWEN_BACKENDS = ("internvl3", "gemma3")
GROUNDING_CONDITIONS = ("real", "none", "caption")


def nonqwen_runtime_metadata_valid(metadata: Any, backend: str) -> bool:
    """True when a recorded runtime block matches what this backend must report."""
    if not isinstance(metadata, dict) or metadata.get("backend") != backend:
        return False
    if metadata.get("generation_callable") is not True:
        return False
    if backend == "internvl3":
        base_valid = (
            isinstance(metadata.get("generation_shim_applied"), bool)
            and metadata.get("generation_config_ready") is True
            and metadata.get("legacy_cache_only") is True
            and metadata.get("timm_version") == "0.9.12"
            and metadata.get("use_flash_attn") is False
        )
        allocation = metadata.get("dynamic_patch_allocation")
        if allocation is None:
            # Single-image rows written before the multi-image patch budget existed carry
            # no allocation field.
            return base_valid
        return (
            base_valid
            and allocation == "balanced_across_images"
            and metadata.get("max_total_dynamic_patches") == 12
        )
    if backend == "gemma3":
        torch_version = metadata.get("torch_version")
        return (
            metadata.get("processor_use_fast") is False
            and isinstance(torch_version, str)
            and gemma3_torch_runtime_supported(torch_version)
        )
    return False


def gemma3_torch_runtime_supported(torch_version: str) -> bool:
    return Version(torch_version.split("+", maxsplit=1)[0]) >= Version("2.6")


def ensure_internvl_generation_compatibility(
    model: Any,
    *,
    generation_mixin_class: type[Any] | None = None,
    generation_config_class: type[Any] | None = None,
) -> bool:
    """Give the InternVL language model a callable `generate` and legacy caches, which some
    released revisions lack. Returns whether anything had to be patched."""
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        raise ValueError("InternVL model has no language_model")
    needs_generation_mixin = not callable(getattr(language_model, "generate", None))
    supports_dynamic_cache = getattr(
        language_model, "_supports_default_dynamic_cache", None
    )
    needs_legacy_cache_override = needs_generation_mixin or (
        callable(supports_dynamic_cache) and bool(supports_dynamic_cache())
    )
    shim_applied = needs_generation_mixin or needs_legacy_cache_override
    if shim_applied:
        if generation_mixin_class is None:
            from transformers.generation import GenerationMixin

            generation_mixin_class = GenerationMixin
        original_class = type(language_model)
        bases = (
            (original_class, generation_mixin_class)
            if needs_generation_mixin
            else (original_class,)
        )
        compatible_class = type(
            f"{original_class.__name__}WithGeneration",
            bases,
            {
                "__module__": original_class.__module__,
                "_supports_default_dynamic_cache": classmethod(
                    lambda cls: False
                ),
            },
        )
        language_model.__class__ = compatible_class
    if not callable(getattr(language_model, "generate", None)):
        raise RuntimeError("InternVL generation compatibility repair did not expose generate")
    cache_support = getattr(language_model, "_supports_default_dynamic_cache", None)
    if not callable(cache_support) or cache_support() is not False:
        raise RuntimeError("InternVL compatibility repair did not select legacy caches")
    if getattr(language_model, "generation_config", None) is None:
        model_config = getattr(language_model, "config", None)
        if model_config is None:
            raise ValueError("InternVL language_model lacks config for generation")
        if generation_config_class is None:
            from transformers import GenerationConfig

            generation_config_class = GenerationConfig
        language_model.generation_config = generation_config_class.from_model_config(
            model_config
        )
    return shim_applied


def validate_content(content: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(content):
        item_type = str(item.get("type", ""))
        if item_type == "image":
            value = str(item.get("image", ""))
            if not value:
                raise ValueError(f"empty image path at content index {index}")
            normalized.append({"type": "image", "image": value})
        elif item_type == "text":
            value = str(item.get("text", ""))
            if not value:
                raise ValueError(f"empty text at content index {index}")
            normalized.append({"type": "text", "text": value})
        else:
            raise ValueError(f"unsupported content type at index {index}: {item_type!r}")
    if not normalized or not any(item["type"] == "text" for item in normalized):
        raise ValueError("VLM content requires at least one text part")
    return normalized


def internvl_question(content: Iterable[dict[str, Any]]) -> tuple[str, list[str]]:
    normalized = validate_content(content)
    image_count = sum(item["type"] == "image" for item in normalized)
    image_paths: list[str] = []
    parts: list[str] = []
    for item in normalized:
        if item["type"] == "image":
            image_paths.append(item["image"])
            marker = (
                "<image>\n"
                if image_count == 1
                else f"Image-{len(image_paths)}: <image>\n"
            )
            parts.append(marker)
        else:
            parts.append(item["text"])
    return "".join(parts), image_paths


def gemma_messages(content: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"role": "user", "content": validate_content(content)}]


def caption_qa_prompt(caption: str, question: str) -> str:
    caption = caption.strip()
    if not caption:
        raise ValueError("fixed question-blind caption cannot be empty")
    return (
        "Answer the question using only the image caption. "
        "If the caption does not contain enough information, make the best possible "
        "answer from the caption.\n"
        f"Caption: {caption}\n"
        f"Question: {question.strip()}\n"
        f"{ANSWER_FORMAT_CONTRACT}"
    )


def grounding_content(
    row: dict[str, Any],
    side: str,
    condition: str,
    caption_row: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Message content for one member of a grounding pair under one image condition."""
    if side not in {"a", "b"}:
        raise ValueError(f"invalid pair side: {side}")
    if condition not in GROUNDING_CONDITIONS:
        raise ValueError(f"unsupported image condition: {condition}")
    question = str(row["question"])
    if condition == "real":
        return [
            {"type": "image", "image": str(resolve_data_reference(row[f"image_{side}_path"]))},
            {"type": "text", "text": format_question(question)},
        ]
    if condition == "none":
        return [{"type": "text", "text": format_question(question)}]
    if caption_row is None:
        raise ValueError(f"caption condition lacks pair {row['pair_id']}")
    if str(caption_row.get("pair_id")) != str(row["pair_id"]):
        raise ValueError("caption row pair identity mismatch")
    caption = str(caption_row.get(f"caption_{side}", ""))
    return [{"type": "text", "text": caption_qa_prompt(caption, question)}]


def _dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """InternVL's own tiling: the closest tile grid by aspect ratio, plus a thumbnail."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    best_ratio = (1, 1)
    best_difference = float("inf")
    area = width * height
    for ratio in target_ratios:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference or (
            difference == best_difference
            and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
        ):
            best_difference = difference
            best_ratio = ratio
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    tiles = []
    columns = target_width // image_size
    for index in range(best_ratio[0] * best_ratio[1]):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size), Image.Resampling.BICUBIC))
    return tiles


def internvl_patch_budgets(image_count: int, total_budget: int) -> list[int]:
    """Allocate a strict total patch budget across all images in one request."""
    if image_count < 0 or total_budget < 1:
        raise ValueError("image count must be nonnegative and patch budget positive")
    if image_count == 0:
        return []
    if image_count > total_budget:
        raise ValueError(
            f"cannot allocate at least one patch to {image_count} images "
            f"from total budget {total_budget}"
        )
    quotient, remainder = divmod(total_budget, image_count)
    return [quotient + int(index < remainder) for index in range(image_count)]


def _dynamic_preprocess_capped(
    image: Image.Image, *, patch_budget: int, image_size: int = 448
) -> list[Image.Image]:
    if patch_budget < 1:
        raise ValueError("per-image patch budget must be positive")
    tiles = _dynamic_preprocess(
        image,
        max_num=patch_budget,
        image_size=image_size,
        use_thumbnail=patch_budget > 1,
    )
    if len(tiles) <= patch_budget:
        return tiles
    # The dynamic tiler appends the thumbnail after the grid. Preserve it while
    # enforcing the request-level memory bound.
    return tiles[: patch_budget - 1] + [tiles[-1]]


@dataclass
class InternVL3Adapter:
    model_path: str
    device: str = "cuda"
    max_new_tokens: int = 384
    max_dynamic_patches: int = 12
    _model: Any = None
    _tokenizer: Any = None
    _transform: Any = None
    _generation_shim_applied: bool = False

    def runtime_metadata(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("InternVL runtime metadata requires a loaded model")
        import timm

        return {
            "backend": "internvl3",
            "generation_callable": callable(
                getattr(self._model.language_model, "generate", None)
            ),
            "generation_shim_applied": self._generation_shim_applied,
            "generation_config_ready": self._model.language_model.generation_config
            is not None,
            "legacy_cache_only": self._model.language_model._supports_default_dynamic_cache()
            is False,
            "timm_version": timm.__version__,
            "use_flash_attn": False,
            "max_total_dynamic_patches": self.max_dynamic_patches,
            "dynamic_patch_allocation": "balanced_across_images",
        }

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import torchvision.transforms as transforms
        from transformers import AutoModel, AutoTokenizer

        self._model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            local_files_only=True,
        )
        self._generation_shim_applied = ensure_internvl_generation_compatibility(
            self._model
        )
        self._model.eval()
        self._model.to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True,
        )
        self._transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda image: image.convert("RGB") if image.mode != "RGB" else image
                ),
                transforms.Resize(
                    (448, 448), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def generate(self, content: Iterable[dict[str, Any]]) -> str:
        self.load()
        import torch

        question, image_paths = internvl_question(content)
        pixel_values = None
        patch_counts: list[int] = []
        if image_paths:
            tensors = []
            patch_budgets = internvl_patch_budgets(
                len(image_paths), self.max_dynamic_patches
            )
            for image_path, patch_budget in zip(image_paths, patch_budgets):
                with Image.open(Path(image_path)) as image:
                    tiles = _dynamic_preprocess_capped(
                        image.convert("RGB"), patch_budget=patch_budget
                    )
                image_tensor = torch.stack([self._transform(tile) for tile in tiles])
                tensors.append(image_tensor)
                patch_counts.append(int(image_tensor.shape[0]))
            if sum(patch_counts) > self.max_dynamic_patches:
                raise AssertionError("InternVL request exceeded its total patch budget")
            pixel_values = torch.cat(tensors).to(device=self.device, dtype=torch.bfloat16)
        generation_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        kwargs: dict[str, Any] = {}
        if len(image_paths) > 1:
            kwargs["num_patches_list"] = patch_counts
        with torch.inference_mode():
            response = self._model.chat(
                self._tokenizer,
                pixel_values,
                question,
                generation_config,
                **kwargs,
            )
        return str(response).strip()


@dataclass
class Gemma3Adapter:
    model_path: str
    device: str = "cuda"
    max_new_tokens: int = 384
    _model: Any = None
    _processor: Any = None

    def runtime_metadata(self) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Gemma runtime metadata requires a loaded model")
        import torch

        return {
            "backend": "gemma3",
            "generation_callable": callable(getattr(self._model, "generate", None)),
            "processor_use_fast": False,
            "torch_version": torch.__version__,
        }

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        if not gemma3_torch_runtime_supported(torch.__version__):
            raise RuntimeError(
                "Gemma-3 multimodal inference requires torch>=2.6 for visual-token "
                f"mask composition; found {torch.__version__}"
            )

        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            local_files_only=True,
        ).eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )

    def generate(self, content: Iterable[dict[str, Any]]) -> str:
        self.load()
        import torch

        messages = gemma_messages(content)
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output[0][input_length:]
        return self._processor.decode(generated, skip_special_tokens=True).strip()


def create_nonqwen_adapter(
    backend: str,
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 384,
) -> InternVL3Adapter | Gemma3Adapter:
    if backend == "internvl3":
        return InternVL3Adapter(model_path, device, max_new_tokens)
    if backend == "gemma3":
        return Gemma3Adapter(model_path, device, max_new_tokens)
    raise ValueError(f"unsupported non-Qwen backend: {backend}")
