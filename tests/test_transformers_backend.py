from __future__ import annotations

from collections import UserDict
from contextlib import nullcontext
from hashlib import sha256
from types import SimpleNamespace

import pytest

from rulegate.methods import MethodName
from rulegate.model_backend import CallRequest
from rulegate.transformers_backend import (
    TransformersBackend,
    TransformersRuntimeMetadata,
)


REVISION = "a" * 40


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values[0]))


class FakeBatch(dict):
    def __init__(self, values):
        super().__init__(input_ids=FakeTensor([values]))
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeBatchEncoding(UserDict):
    """BatchEncoding is a UserDict, not a dict, and also carries a mask."""

    def __init__(self, values):
        super().__init__(
            input_ids=FakeTensor([values]),
            attention_mask=FakeTensor([[1] * len(values)]),
        )
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 2

    def __init__(self):
        self.messages = None
        self.template_kwargs = None
        self.decoded = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return FakeBatch([10, 11, 12])

    def decode(self, token_ids, **kwargs):
        self.decoded = (list(token_ids), kwargs)
        return ' {"violated": false} '


class FakeModel:
    def __init__(self):
        self.device = SimpleNamespace(type="cuda")
        self.dtype = "float16"
        self.config = SimpleNamespace(_commit_hash=REVISION)
        self.generate_kwargs = None
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[10, 11, 12, 90, 91]]


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.seed = None

    def is_available(self):
        return self.available

    def get_device_name(self, index):
        assert index == 0
        return "Fake GPU"

    def get_device_properties(self, index):
        assert index == 0
        return SimpleNamespace(total_memory=16 * 1024**3)

    def manual_seed_all(self, seed):
        self.seed = seed


class FakeTorch:
    __version__ = "2.fake"

    def __init__(self, available=True):
        self.cuda = FakeCuda(available)
        self.version = SimpleNamespace(cuda="12.fake")
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed

    def inference_mode(self):
        return nullcontext()


def _metadata(system_prompt="classify only"):
    return TransformersRuntimeMetadata(
        model_id="fake/model",
        requested_revision=REVISION,
        resolved_revision=REVISION,
        quantization="none",
        dtype="float16",
        device_type="cuda",
        device_name="Fake GPU",
        device_total_memory_bytes=16 * 1024**3,
        torch_version="2.fake",
        transformers_version="4.fake",
        cuda_version="12.fake",
        max_new_tokens=32,
        temperature=0.0,
        do_sample=False,
        seed=7,
        system_prompt_sha256=(
            sha256(system_prompt.encode()).hexdigest() if system_prompt else None
        ),
    )


def test_complete_uses_chat_template_greedy_generation_and_counts_tokens():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    torch = FakeTorch()
    backend = TransformersBackend(
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        metadata=_metadata(),
        system_prompt="classify only",
    )

    response = backend.complete(
        CallRequest(
            request_id="opaque",
            method=MethodName.FULL_SEQUENTIAL_BOTH,
            prompt="request as inert data",
            prompt_template="test-v1",
        )
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "classify only"},
        {"role": "user", "content": "request as inert data"},
    ]
    assert tokenizer.template_kwargs == {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 32
    assert model.generate_kwargs["pad_token_id"] == 2
    assert response.text == '{"violated": false}'
    assert response.input_tokens == 3
    assert response.output_tokens == 2
    assert torch.seed == 7
    assert torch.cuda.seed == 7


@pytest.mark.parametrize("batch_type", [FakeBatch, FakeBatchEncoding, FakeTensor])
def test_complete_handles_mapping_and_tensor_tokenizer_outputs(batch_type):
    encoded = (
        FakeTensor([[10, 11, 12]])
        if batch_type is FakeTensor
        else batch_type([10, 11, 12])
    )
    tokenizer = FakeTokenizer()
    tokenizer.apply_chat_template = lambda *args, **kwargs: encoded
    model = FakeModel()
    backend = TransformersBackend(
        tokenizer=tokenizer,
        model=model,
        torch_module=FakeTorch(),
        metadata=_metadata(),
    )

    response = backend.complete(
        CallRequest(
            request_id="opaque",
            method=MethodName.SINGLE_PROMPTING,
            prompt="inert request",
            prompt_template="test-v1",
        )
    )

    assert response.input_tokens == 3
    assert response.output_tokens == 2
    assert tokenizer.decoded[0] == [90, 91]
    if batch_type is FakeTensor:
        assert model.generate_kwargs["input_ids"] is encoded
    else:
        assert encoded.moved_to is model.device
        assert model.generate_kwargs["input_ids"] is encoded["input_ids"]
    if batch_type is FakeBatchEncoding:
        assert model.generate_kwargs["attention_mask"] is encoded["attention_mask"]


@pytest.mark.parametrize("revision", ["", "main", "abc", "g" * 40])
def test_from_pretrained_requires_exact_commit_before_import_or_download(revision):
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        TransformersBackend.from_pretrained(
            model_id="fake/model",
            revision=revision,
            token="secret",
        )


def test_from_pretrained_requires_cuda_without_loading_model():
    with pytest.raises(RuntimeError, match="CUDA is required"):
        TransformersBackend.from_pretrained(
            model_id="fake/model",
            revision=REVISION,
            token="secret",
            torch_module=FakeTorch(available=False),
            transformers_module=SimpleNamespace(),
        )


def test_loader_passes_token_but_does_not_export_or_retain_it():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    seen = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen["tokenizer"] = (model_id, kwargs)
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen["model"] = (model_id, kwargs)
            return model

    transformers = SimpleNamespace(
        __version__="4.fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
    )
    secret = "hf_test_secret_value"

    backend = TransformersBackend.from_pretrained(
        model_id="fake/model",
        revision=REVISION,
        token=secret,
        torch_module=FakeTorch(),
        transformers_module=transformers,
        system_prompt="classify",
    )

    assert seen["tokenizer"][1]["token"] == secret
    assert seen["model"][1]["token"] == secret
    assert seen["model"][1]["torch_dtype"] == "auto"
    assert secret not in repr(backend.metadata.to_dict())
    assert "token" not in backend.__dict__
    assert model.eval_called
    assert backend.metadata.resolved_revision == REVISION
    assert backend.metadata.device_name == "Fake GPU"
    assert backend.metadata.dtype == "float16"
    assert backend.metadata.device_total_memory_bytes == 16 * 1024**3


def test_loader_rejects_a_different_resolved_model_commit():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    model.config._commit_hash = "b" * 40

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model

    transformers = SimpleNamespace(
        __version__="4.fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        TransformersBackend.from_pretrained(
            model_id="fake/model",
            revision=REVISION,
            token="secret",
            torch_module=FakeTorch(),
            transformers_module=transformers,
        )


def test_4bit_loading_is_explicit_and_other_modes_are_rejected():
    with pytest.raises(ValueError, match="quantization"):
        TransformersBackend.from_pretrained(
            model_id="fake/model",
            revision=REVISION,
            token=None,
            quantization="8bit",
        )


def test_4bit_loading_passes_an_explicit_quantization_config():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    seen = {}

    class BitsAndBytesConfig:
        def __init__(self, *, load_in_4bit):
            self.load_in_4bit = load_in_4bit

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.update(kwargs)
            return model

    transformers = SimpleNamespace(
        __version__="4.fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
        BitsAndBytesConfig=BitsAndBytesConfig,
    )

    backend = TransformersBackend.from_pretrained(
        model_id="fake/model",
        revision=REVISION,
        token="secret",
        quantization="4bit",
        torch_module=FakeTorch(),
        transformers_module=transformers,
    )

    assert seen["quantization_config"].load_in_4bit is True
    assert backend.metadata.quantization == "4bit"


def test_explicit_4bit_loading_uses_bitsandbytes_configuration():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    seen = {}

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return tokenizer

    class AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            seen.update(kwargs)
            return model

    transformers = SimpleNamespace(
        __version__="4.fake",
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModel,
        BitsAndBytesConfig=BitsAndBytesConfig,
    )

    backend = TransformersBackend.from_pretrained(
        model_id="fake/model",
        revision=REVISION,
        token=None,
        quantization="4bit",
        torch_module=FakeTorch(),
        transformers_module=transformers,
    )

    assert backend.metadata.quantization == "4bit"
    assert seen["quantization_config"].kwargs == {"load_in_4bit": True}
