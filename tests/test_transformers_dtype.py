from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from rulegate.transformers_backend import TransformersBackend


REVISION = "a" * 40


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2


class FakeModel:
    def __init__(self):
        self.device = SimpleNamespace(type="cuda")
        self.dtype = "torch.float16"
        self.config = SimpleNamespace(_commit_hash=REVISION)
        self.hf_device_map = {"model.embed_tokens": 0, "lm_head": 1}

    def eval(self):
        return None


class FakeT4Cuda:
    def is_available(self):
        return True

    def device_count(self):
        return 2

    def get_device_name(self, index):
        return "Tesla T4"

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=15 * 1024**3)

    def get_device_capability(self, index):
        return (7, 5)

    def memory_allocated(self, index):
        return 8 * 1024**3

    def manual_seed_all(self, seed):
        return None


class FakeT4Torch:
    __version__ = "2.fake"
    float16 = "torch.float16"
    bfloat16 = "torch.bfloat16"

    def __init__(self):
        self.cuda = FakeT4Cuda()
        self.version = SimpleNamespace(cuda="12.fake")

    def manual_seed(self, seed):
        return None

    def inference_mode(self):
        return nullcontext()


def _fake_transformers(model, seen):
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.update(kwargs)
            return model

    return SimpleNamespace(
        __version__="4.fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
    )


def test_float16_is_explicit_and_multi_gpu_metadata_is_recorded():
    seen: dict = {}
    backend = TransformersBackend.from_pretrained(
        model_id="fake/model",
        revision=REVISION,
        token="secret-token-value",
        dtype="float16",
        torch_module=FakeT4Torch(),
        transformers_module=_fake_transformers(FakeModel(), seen),
    )

    assert seen["torch_dtype"] == "torch.float16"
    metadata = backend.metadata.to_dict()
    assert metadata["requested_dtype"] == "float16"
    assert [d["compute_capability"] for d in metadata["cuda_devices"]] == ["7.5", "7.5"]
    assert metadata["device_map"] == (("model.embed_tokens", "0"), ("lm_head", "1"))
    assert metadata["memory_allocated_bytes_after_load"] == (8 * 1024**3, 8 * 1024**3)
    assert metadata["load_seconds"] >= 0
    assert "secret-token-value" not in repr(metadata)


def test_bfloat16_is_rejected_on_gpus_without_native_support():
    seen: dict = {}
    with pytest.raises(RuntimeError, match="compute capability >= 8.0"):
        TransformersBackend.from_pretrained(
            model_id="fake/model",
            revision=REVISION,
            token="secret",
            dtype="bfloat16",
            torch_module=FakeT4Torch(),
            transformers_module=_fake_transformers(FakeModel(), seen),
        )
    assert seen == {}


def test_unknown_dtype_is_rejected_before_import():
    with pytest.raises(ValueError, match="dtype"):
        TransformersBackend.from_pretrained(
            model_id="fake/model", revision=REVISION, token=None, dtype="float32"
        )
