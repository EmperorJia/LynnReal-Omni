"""Per-request H3 memory accounting and bounded INT8 kernel launches.

No process-wide backend/reserve switch is made while a prompt executes. Small
INT8 calls keep the selected implementation; large calls use the same arithmetic
on independent row slices before an overflowing kernel can corrupt CUDA state.
"""
from __future__ import annotations

import functools
import logging
import math

import torch

GB = 1024 ** 3
# Leave margin below signed 32-bit *byte* offsets (not just element offsets).
MAX_BUFFER_BYTES = 1 << 30
MAX_CHUNK_ROWS = 16384
_LOGGED = set()


def chunk_rows(rows, input_width, output_width, input_bytes=2, output_bytes=2):
    width_bytes = max(input_width * input_bytes, output_width * output_bytes,
                      output_width * 4)  # eager INT32 accumulator too
    if rows * width_bytes < MAX_BUFFER_BYTES:
        return rows
    return max(1, min(MAX_CHUNK_ROWS, (MAX_BUFFER_BYTES - 1) // width_bytes))


def conditioning_rows(cond):
    """Count the extra packed rows absent from the core H3 memory estimate."""
    refs = 0
    for key in ("minimax_refs", "minimax_keyframes"):
        for block in cond.get(key, ()) or ():
            latent = block.get("latent")
            if latent is not None:
                b, _, t, h, w = latent.shape
                refs += b * t * math.ceil(h / 2) * math.ceil(w / 2)
            audio = block.get("audio_latent")
            if audio is not None:
                refs += audio.shape[0] * audio.shape[-2] * audio.shape[-1]
    text = cond.get("cross_attn")
    text_rows = math.prod(text.shape[:-1]) if text is not None else 0
    return refs, text_rows


def additional_memory(conds):
    rows = max((conditioning_rows(c) for cs in conds.values() for c in cs),
               default=(0, 0), key=lambda pair: sum(pair))
    # Short ordinary prompts retain ComfyUI's existing estimate. Reference
    # requests need working space for their packed rows plus a staging margin.
    extra_rows = rows[0] + max(0, rows[1] - 4096)
    if not extra_rows:
        return 0, rows
    # Conservative working-set allowance, calibrated on 768p H100 reference
    # and pose requests. It is a loading estimate, not an allocation or a cap.
    return 2 * GB + extra_rows * 384 * 1024, rows


def encoder_memory(tokens):
    """Budget expanded Qwen vision/text tokens before the encoder is loaded.

    Image placeholders expand after CLIP.load_model; counting the unexpanded
    token list would miss both the vision MLP and the language-model sequence.
    Match Qwen's 16px patches, 2x2 merge and pixel bounds using shapes only.
    """
    longest, peak_patches = 0, 0
    batches = tokens.get("qwen3vl_32b", ())
    for batch in batches:
        sequence = 0
        for entry in batch:
            token = entry[0]
            if not isinstance(token, dict) or token.get("type") != "image":
                sequence += 1
                continue
            height, width = token["data"].shape[1:3]
            h, w = round(height / 32) * 32, round(width / 32) * 32
            if h * w > 12845056:
                scale = math.sqrt(height * width / 12845056)
                h = max(32, math.floor(height / scale / 32) * 32)
                w = max(32, math.floor(width / scale / 32) * 32)
            elif h * w < 3136:
                scale = math.sqrt(3136 / (height * width))
                h = math.ceil(height * scale / 32) * 32
                w = math.ceil(width * scale / 32) * 32
            patches = (h // 16) * (w // 16)
            peak_patches = max(peak_patches, patches)
            sequence += patches // 4
        longest = max(longest, sequence)
    if not peak_patches and longest <= 4096:
        return 0, longest, peak_patches
    # Qwen runs its embeddings/MLP in FP32: gate, up, activation/product,
    # residuals and retained vision DeepStack features coexist. Allow 512 KiB
    # per expanded row plus staging space. Weights are accounted for separately
    # by ComfyUI. Vision blocks execute serially, so use their peak patch count.
    required = 2 * GB + longest * max(1, len(batches)) * 512 * 1024 + peak_patches * 64 * 1024
    return required, longest, peak_patches


def _bounded_int8(original):
    @functools.wraps(original)
    def int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, **kwargs):
        out_dtype = out_dtype or torch.bfloat16
        rows = math.prod(x.shape[:-1])
        out_bytes = torch.empty((), dtype=out_dtype).element_size()
        limit = chunk_rows(rows, x.shape[-1], weight.shape[0],
                           x.element_size(), out_bytes)
        if rows <= limit:
            return original(x, weight, weight_scale, bias=bias,
                            out_dtype=out_dtype, **kwargs)
        signature = (x.shape[-1], weight.shape[0], limit)
        if signature not in _LOGGED:
            _LOGGED.add(signature)
            logging.info("LynnReal: safe INT8 rows %d -> chunks <= %d (%d -> %d columns); "
                         "retaining the selected backend.", rows, limit, *signature[:2])
        flat = x.reshape(rows, x.shape[-1])
        output = torch.empty((rows, weight.shape[0]), device=x.device,
                             dtype=out_dtype)
        for start in range(0, rows, limit):
            part = original(flat[start:start + limit].contiguous(), weight, weight_scale,
                            bias=bias, out_dtype=out_dtype, **kwargs)
            output[start:start + limit].copy_(part)
        return output.reshape(*x.shape[:-1], weight.shape[0])

    int8_linear._lynnreal_safe_rows = True
    return int8_linear


def install():
    import comfy.model_base
    import comfy.sampler_helpers
    from comfy.text_encoders.minimax import MiniMaxH3TEModel
    try:
        import comfy_kitchen as ck
    except ImportError:
        ck = None

    # QuantizedTensor's normal linears dispatch via torch.ops, while the fused
    # activation path calls ck.int8_linear. Both resolve their implementation
    # through this registry. Patching only ck.int8_linear misses fc1 entirely.
    registry = getattr(ck, "registry", None)
    resolve = getattr(registry, "get_implementation", None)
    if resolve is None:
        # BF16-only installs and the pack's CPU verifier can omit the optional
        # INT8 backend. They still need encoder/sampling memory accounting.
        logging.info("LynnReal: comfy-kitchen INT8 registry unavailable; memory accounting remains active.")
    elif not getattr(resolve, "_lynnreal_safe_rows", False):
        wrappers = {}

        @functools.wraps(resolve)
        def get_implementation(func_name, backend=None, kwargs=None):
            implementation = resolve(func_name, backend=backend, kwargs=kwargs)
            if func_name != "int8_linear":
                return implementation
            if implementation not in wrappers:
                wrappers[implementation] = _bounded_int8(implementation)
            return wrappers[implementation]

        get_implementation._lynnreal_safe_rows = True
        registry.get_implementation = get_implementation

    estimate = comfy.sampler_helpers.estimate_memory
    if not getattr(estimate, "_lynnreal_reference_memory", False):
        @functools.wraps(estimate)
        def estimate_memory(model, noise_shape, conds):
            required, minimum = estimate(model, noise_shape, conds)
            if not isinstance(model.model, comfy.model_base.MiniMaxH3):
                return required, minimum
            extra, (refs, text) = additional_memory(conds)
            if extra:
                logging.info("LynnReal: H3 sampling memory includes %d reference and %d text "
                             "rows: +%.2f GiB, minimum %.2f GiB (request-local).",
                             refs, text, extra / GB, (minimum + extra) / GB)
            return required + extra, minimum + extra

        estimate_memory._lynnreal_reference_memory = True
        comfy.sampler_helpers.estimate_memory = estimate_memory

    encoder_estimate = getattr(MiniMaxH3TEModel, "memory_estimation_function", None)
    if not getattr(encoder_estimate, "_lynnreal_encoder_memory", False):
        def memory_estimation_function(self, tokens, device=None):
            baseline = encoder_estimate(self, tokens, device=device) if encoder_estimate else 0
            required, sequence, patches = encoder_memory(tokens)
            if required:
                logging.info("LynnReal: H3 encoder memory for %d expanded tokens and %d peak "
                             "vision patches: %.2f GiB (request-local).",
                             sequence, patches, max(baseline, required) / GB)
            return max(baseline, required)

        memory_estimation_function._lynnreal_encoder_memory = True
        MiniMaxH3TEModel.memory_estimation_function = memory_estimation_function
    logging.info("LynnReal: automatic H3 encoder/conditioning memory and INT8 row safety installed.")
