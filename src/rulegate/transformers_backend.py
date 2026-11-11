"""Optional Hugging Face backend for the Stage 2 Kaggle runtime.

The module has no import-time dependency on PyTorch or Transformers.  Real model
loading is explicit, requires a CUDA runtime and an immutable 40-character model
commit, and never stores the Hugging Face token in exported metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib
from importlib import metadata as importlib_metadata
import re
from time import perf_counter
from typing import Any

from .model_backend import CallRequest, RawBackendResponse


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_QUANTIZATIONS = {"none", "4bit"}
# "auto" keeps the checkpoint's stored dtype (bfloat16 for Llama-3.1). Kaggle's
# T4 GPUs (compute capability 7.5) have no native bfloat16, so the T4 profile
# selects float16 explicitly. The choice is recorded and must be identical
# across every compared condition.
_DTYPES = {"auto", "float16", "bfloat16"}


@dataclass(frozen=True, slots=True)
class CudaDeviceInfo:
    index: int
    name: str
    total_memory_bytes: int
    compute_capability: str | None


@dataclass(frozen=True, slots=True)
class TransformersRuntimeMetadata:
    model_id: str
    requested_revision: str
    resolved_revision: str
    quantization: str
    dtype: str
    device_type: str
    device_name: str
    device_total_memory_bytes: int
    torch_version: str
    transformers_version: str
    cuda_version: str | None
    max_new_tokens: int
    temperature: float
    do_sample: bool
    seed: int
    system_prompt_sha256: str | None
    requested_dtype: str = "auto"
    cuda_devices: tuple[CudaDeviceInfo, ...] = ()
    device_map: tuple[tuple[str, str], ...] = ()
    accelerate_version: str | None = None
    bitsandbytes_version: str | None = None
    load_seconds: float | None = None
    memory_allocated_bytes_after_load: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return safe run metadata; credentials are deliberately absent."""

        return asdict(self)


def _validated_dtype(dtype: str | None) -> str:
    normalized = "auto" if dtype is None else str(dtype).lower()
    if normalized not in _DTYPES:
        raise ValueError("dtype must be 'auto', 'float16', or 'bfloat16'")
    return normalized


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cuda_devices(torch: Any) -> tuple[CudaDeviceInfo, ...]:
    cuda = torch.cuda
    count_fn = getattr(cuda, "device_count", None)
    count = int(count_fn()) if callable(count_fn) else 1
    devices: list[CudaDeviceInfo] = []
    for index in range(count):
        properties = cuda.get_device_properties(index)
        capability_fn = getattr(cuda, "get_device_capability", None)
        capability = None
        if callable(capability_fn):
            major, minor = capability_fn(index)
            capability = f"{major}.{minor}"
        devices.append(
            CudaDeviceInfo(
                index=index,
                name=str(cuda.get_device_name(index)),
                total_memory_bytes=int(properties.total_memory),
                compute_capability=capability,
            )
        )
    return tuple(devices)


def _memory_allocated(torch: Any, device_count: int) -> tuple[int, ...]:
    allocated_fn = getattr(torch.cuda, "memory_allocated", None)
    if not callable(allocated_fn):
        return ()
    return tuple(int(allocated_fn(index)) for index in range(device_count))


def _validated_revision(revision: str) -> str:
    if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
        raise ValueError(
            "revision must be an exact 40-character hexadecimal Hugging Face commit"
        )
    return revision.lower()


def _validated_quantization(quantization: str | None) -> str:
    normalized = "none" if quantization is None else quantization.lower()
    if normalized not in _QUANTIZATIONS:
        raise ValueError("quantization must be 'none', None, or '4bit'")
    return normalized


class TransformersBackend:
    """Greedy chat-completion backend implementing ``RawBackend``.

    ``tokenizer``, ``model`` and ``torch_module`` injection keeps CPU tests fully
    offline. Production callers should use :meth:`from_pretrained`.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        metadata: TransformersRuntimeMetadata,
        system_prompt: str | None = None,
    ) -> None:
        if metadata.temperature != 0.0 or metadata.do_sample:
            raise ValueError("RuleGate requires temperature=0 and do_sample=False")
        if metadata.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.tokenizer = tokenizer
        self.model = model
        self._torch = torch_module
        self.metadata = metadata
        self.system_prompt = system_prompt

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        token: str | None,
        quantization: str | None = None,
        dtype: str | None = "auto",
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
        seed: int = 484654,
        torch_module: Any | None = None,
        transformers_module: Any | None = None,
    ) -> "TransformersBackend":
        """Load a pinned gated model after validating the actual CUDA runtime.

        The injectable module arguments are for offline tests.  The authentication
        token is passed directly to Hugging Face loaders and is not assigned to
        the backend or its metadata.
        """

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty")
        requested_revision = _validated_revision(revision)
        normalized_quantization = _validated_quantization(quantization)
        requested_dtype = _validated_dtype(dtype)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        torch = torch_module or importlib.import_module("torch")
        transformers = transformers_module or importlib.import_module("transformers")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Stage 2 Transformers backend")
        cuda_devices = _cuda_devices(torch)
        if requested_dtype == "bfloat16":
            unsupported = [
                device
                for device in cuda_devices
                if device.compute_capability is not None
                and tuple(int(part) for part in device.compute_capability.split("."))
                < (8, 0)
            ]
            if unsupported:
                raise RuntimeError(
                    "bfloat16 requires compute capability >= 8.0; attached GPU "
                    f"{unsupported[0].name} reports {unsupported[0].compute_capability}"
                )

        started = perf_counter()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=requested_revision,
            token=token,
        )
        model_kwargs: dict[str, object] = {
            "revision": requested_revision,
            "token": token,
            "device_map": "auto",
            "torch_dtype": (
                "auto" if requested_dtype == "auto" else getattr(torch, requested_dtype)
            ),
        }
        if normalized_quantization == "4bit":
            model_kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True
            )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        model.eval()
        load_seconds = perf_counter() - started
        raw_device_map = getattr(model, "hf_device_map", None)
        device_map = (
            tuple((str(name), str(device)) for name, device in raw_device_map.items())
            if isinstance(raw_device_map, Mapping)
            else ()
        )

        resolved_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
        if isinstance(resolved_revision, str) and _COMMIT_RE.fullmatch(
            resolved_revision
        ):
            resolved_revision = resolved_revision.lower()
            if resolved_revision != requested_revision:
                raise RuntimeError(
                    "loaded model commit does not match the requested revision"
                )
        else:
            resolved_revision = requested_revision
        device = getattr(model, "device", None)
        device_type = str(getattr(device, "type", "cuda"))
        device_name = str(torch.cuda.get_device_name(0))
        device_properties = torch.cuda.get_device_properties(0)
        device_total_memory_bytes = int(device_properties.total_memory)
        prompt_hash = (
            sha256(system_prompt.encode("utf-8")).hexdigest()
            if system_prompt is not None
            else None
        )
        cuda_value = getattr(getattr(torch, "version", None), "cuda", None)
        metadata = TransformersRuntimeMetadata(
            model_id=model_id,
            requested_revision=requested_revision,
            resolved_revision=resolved_revision,
            quantization=normalized_quantization,
            dtype=str(getattr(model, "dtype", "unknown")),
            device_type=device_type,
            device_name=device_name,
            device_total_memory_bytes=device_total_memory_bytes,
            torch_version=str(getattr(torch, "__version__", "unknown")),
            transformers_version=str(
                getattr(transformers, "__version__", "unknown")
            ),
            cuda_version=str(cuda_value) if cuda_value else None,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            seed=seed,
            system_prompt_sha256=prompt_hash,
            requested_dtype=requested_dtype,
            cuda_devices=cuda_devices,
            device_map=device_map,
            accelerate_version=_distribution_version("accelerate"),
            bitsandbytes_version=_distribution_version("bitsandbytes"),
            load_seconds=load_seconds,
            memory_allocated_bytes_after_load=_memory_allocated(
                torch, len(cuda_devices)
            ),
        )
        return cls(
            tokenizer=tokenizer,
            model=model,
            torch_module=torch,
            metadata=metadata,
            system_prompt=system_prompt,
        )

    def complete(self, request: CallRequest) -> RawBackendResponse:
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise ValueError("prompt must be non-empty")
        messages: list[dict[str, str]] = []
        system_prompt = (
            request.system_prompt
            if request.system_prompt is not None
            else self.system_prompt
        )
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        model_device = getattr(self.model, "device", None)
        if hasattr(encoded, "to") and model_device is not None:
            encoded = encoded.to(model_device)
        # Hugging Face BatchEncoding implements Mapping via UserDict.
        # Treating it as a tensor fails on .shape and loses attention_mask.
        if isinstance(encoded, Mapping):
            model_inputs = encoded
            input_ids = encoded["input_ids"]
        else:
            model_inputs = {"input_ids": encoded}
            input_ids = encoded
        input_tokens = int(input_ids.shape[-1])

        self._torch.manual_seed(self.metadata.seed)
        if hasattr(self._torch.cuda, "manual_seed_all"):
            self._torch.cuda.manual_seed_all(self.metadata.seed)
        inference_context = (
            self._torch.inference_mode()
            if hasattr(self._torch, "inference_mode")
            else nullcontext()
        )
        generation_kwargs: dict[str, object] = {
            "max_new_tokens": self.metadata.max_new_tokens,
            "do_sample": False,
        }
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        with inference_context:
            generated = self.model.generate(**model_inputs, **generation_kwargs)
        generated_ids = generated[0][input_tokens:]
        output_tokens = len(generated_ids)
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        return RawBackendResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
