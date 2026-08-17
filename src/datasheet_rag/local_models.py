"""Local chat/vision backends, exposed as a Bedrock-compatible shim.

The five Claude callers in this project — figure description and table-structure
repair (vision), and title inference, chunk summarization, and eval question
generation (text) — build the *same* Anthropic ``messages`` request and call::

    response = client.invoke_model(modelId=..., body=<json>, ...)
    payload = json.loads(response["body"].read())
    text = payload["content"][0]["text"]   # + payload["usage"]

Two local clients mimic that boto3 ``bedrock-runtime`` surface so callers need
only swap which client they construct (via :func:`get_chat_client`):

* :class:`OllamaInvokeClient` — routes to a local Ollama server (HTTP, GGUF).
* :class:`TransformersChatClient` — runs a HuggingFace model in-process via
  PyTorch/CUDA (VLM for vision, causal LM for text). Full precision (or 4-bit),
  precise VRAM control, and supports any HF model id — the only way large VLMs
  fit a consumer GPU.

:func:`get_chat_client` picks per capability (``kind`` = "vision" | "text"):
backend (``vision_backend`` / ``text_backend`` → bedrock vs local) then, for
local, the runtime (``local_vision_runtime`` / ``local_text_runtime`` →
huggingface vs ollama). The incoming Bedrock ``modelId`` is ignored locally.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any, Literal

from datasheet_rag.config import get_settings

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient


def _import_httpx() -> Any:
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "The Ollama backend needs 'httpx' (part of the base install). "
            "Reinstall:  pip install datasheet-rag"
        ) from exc
    return httpx


# ---------------------------------------------------------------------------
# Shared response shaping
# ---------------------------------------------------------------------------


class _ResponseBody:
    """Minimal stand-in for botocore's StreamingBody (only ``.read()``)."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def _anthropic_response(text: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    """Wrap generated text in the Anthropic response shape callers expect."""
    payload = {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": int(input_tokens), "output_tokens": int(output_tokens)},
    }
    return {"body": _ResponseBody(json.dumps(payload).encode("utf-8"))}


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


def _anthropic_to_ollama_messages(
    body: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Translate an Anthropic ``messages`` body into Ollama chat messages.

    Returns ``(messages, has_image)``. Anthropic ``content`` may be a string or
    a list of ``{"type": "text"|"image", ...}`` blocks; Ollama wants text in
    ``content`` and base64 images in a sibling ``images`` array.
    """
    messages: list[dict[str, Any]] = []
    has_image = False

    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        images: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "image":
                data = block.get("source", {}).get("data", "")
                if data:
                    images.append(data)
                    has_image = True

        ollama_msg: dict[str, Any] = {
            "role": role,
            "content": "\n\n".join(t for t in text_parts if t),
        }
        if images:
            ollama_msg["images"] = images
        messages.append(ollama_msg)

    return messages, has_image


class OllamaInvokeClient:
    """``invoke_model`` drop-in that routes to a local Ollama server.

    ``model`` pins the model to use; if omitted, the model is chosen per call
    from message content (vision model when an image is present, else text).
    """

    # Generous read timeout: vision + large num_predict can exceed a minute.
    _TIMEOUT_SECONDS = 600.0

    def __init__(
        self,
        *,
        model: str | None = None,
        host: str | None = None,
        chat_model: str | None = None,
        vision_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model
        self.chat_model = chat_model or settings.local_text_model
        self.vision_model = vision_model or settings.local_vision_model

    def invoke_model(
        self,
        *,
        modelId: str | None = None,  # noqa: N803 - matches boto3 kwarg
        body: str | bytes,
        contentType: str | None = None,  # noqa: N803 - boto3 kwarg, unused
        accept: str | None = None,  # boto3 kwarg, unused
    ) -> dict[str, Any]:
        parsed = json.loads(body)
        messages, has_image = _anthropic_to_ollama_messages(parsed)
        model = self.model or (self.vision_model if has_image else self.chat_model)

        options: dict[str, Any] = {}
        if "max_tokens" in parsed:
            options["num_predict"] = parsed["max_tokens"]
        if "temperature" in parsed:
            options["temperature"] = parsed["temperature"]

        request: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if options:
            request["options"] = options

        httpx = _import_httpx()
        resp = httpx.post(f"{self.host}/api/chat", json=request, timeout=self._TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message", {}) or {}).get("content", "") or ""
        return _anthropic_response(
            text,
            int(data.get("prompt_eval_count", 0) or 0),
            int(data.get("eval_count", 0) or 0),
        )


# ---------------------------------------------------------------------------
# HuggingFace (transformers) client — in-process VLM / causal LM
# ---------------------------------------------------------------------------


_HF_HINT = (
    "The huggingface chat/vision runtime needs 'transformers', 'torch', "
    "'accelerate' (and 'bitsandbytes' for 4-bit). Install:  "
    "pip install 'datasheet-rag[local-hf]'"
)


def _decode_image(b64: str) -> Any:
    import io

    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class TransformersChatClient:
    """``invoke_model`` drop-in that runs a HuggingFace model in-process.

    ``is_vision`` selects an image-text-to-text model (VLM) vs a causal LM. The
    model is loaded lazily on first call (downloading weights from the HF Hub if
    not cached). Optionally 4-bit quantized (bitsandbytes) to fit larger models.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        is_vision: bool = False,
        load_4bit: bool | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self.is_vision = is_vision
        self.model_id = model or (
            settings.local_vision_model if is_vision else settings.local_text_model
        )
        self.load_4bit = settings.local_hf_load_4bit if load_4bit is None else load_4bit
        self.device = device
        self._model: Any = None
        self._proc: Any = None  # processor (vision) or tokenizer (text)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(_HF_HINT) from exc

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
        if self.load_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(_HF_HINT) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = "auto"  # requires accelerate

        if self.is_vision:
            from transformers import AutoModelForImageTextToText

            self._model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
            self._proc = AutoProcessor.from_pretrained(self.model_id)  # type: ignore[no-untyped-call]
        else:
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            self._proc = AutoTokenizer.from_pretrained(self.model_id)

        if not self.load_4bit:  # device_map already placed the 4-bit model
            self._model = self._model.to(device)
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def invoke_model(
        self,
        *,
        modelId: str | None = None,  # noqa: N803 - matches boto3 kwarg
        body: str | bytes,
        contentType: str | None = None,  # noqa: N803 - boto3 kwarg, unused
        accept: str | None = None,  # boto3 kwarg, unused
    ) -> dict[str, Any]:
        self._load()
        import torch

        parsed = json.loads(body)
        max_new = int(parsed.get("max_tokens", 512))
        temperature = float(parsed.get("temperature", 0.0) or 0.0)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new}
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs["do_sample"] = False

        if self.is_vision:
            messages, images = self._hf_vision_messages(parsed)
            prompt = self._proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._proc(
                text=[prompt], images=images or None, return_tensors="pt"
            ).to(self._device)
            in_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)
            gen = out[:, in_len:]
            text = self._proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        else:
            messages = self._hf_text_messages(parsed)
            inputs = self._proc.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self._device)
            in_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)
            gen = out[0][in_len:]
            text = self._proc.decode(gen, skip_special_tokens=True).strip()

        return _anthropic_response(text, in_len, int(gen.shape[-1]))

    @staticmethod
    def _hf_text_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if body.get("system"):
            messages.append({"role": "system", "content": body["system"]})
        for msg in body.get("messages", []):
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = "\n\n".join(
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            messages.append({"role": msg.get("role", "user"), "content": content})
        return messages

    @staticmethod
    def _hf_vision_messages(body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
        messages: list[dict[str, Any]] = []
        images: list[Any] = []
        if body.get("system"):
            sys_content = [{"type": "text", "text": body["system"]}]
            messages.append({"role": "system", "content": sys_content})
        for msg in body.get("messages", []):
            content = msg.get("content", "")
            parts: list[dict[str, Any]] = []
            if isinstance(content, str):
                parts.append({"type": "text", "text": content})
            else:
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image":
                        data = block.get("source", {}).get("data", "")
                        if data:
                            images.append(_decode_image(data))
                            parts.append({"type": "image"})
            messages.append({"role": msg.get("role", "user"), "content": parts})
        return messages, images


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_chat_client(
    *,
    kind: Literal["vision", "text"],
    region: str | None = None,
    profile: str | None = None,
) -> BedrockRuntimeClient | OllamaInvokeClient | TransformersChatClient:
    """Return the client for a chat capability.

    ``kind`` selects the backend (``vision_backend`` vs ``text_backend``); for a
    local backend the runtime (``local_vision_runtime`` / ``local_text_runtime``)
    picks ``huggingface`` (:class:`TransformersChatClient`) or ``ollama``
    (:class:`OllamaInvokeClient`). All return an ``invoke_model``-compatible
    object.
    """
    settings = get_settings()
    is_vision = kind == "vision"
    backend = settings.vision_backend if is_vision else settings.text_backend

    if backend == "local":
        runtime = settings.local_vision_runtime if is_vision else settings.local_text_runtime
        model = settings.local_vision_model if is_vision else settings.local_text_model
        if runtime == "ollama":
            return OllamaInvokeClient(model=model)
        return TransformersChatClient(model=model, is_vision=is_vision)

    from datasheet_rag.embedding.embedder import _bedrock_runtime_client

    return _bedrock_runtime_client(region=region, profile=profile)
