"""Pick the fast comfy-kitchen backend for LynnReal INT8 models, without a launch flag.

`comfy/quant_ops.py` disables comfy-kitchen's CUDA backend on any torch older than cu130 and
leaves the Triton backend off unless `--enable-triton-backend` is passed. With both off, every
quantized op resolves to the eager implementations, and the INT8 GEMM in particular is then a
materialised INT32 accumulator plus a chunked FP32 rescale: the Flash 3-step t2v measures
8.8 s/step that way against 2.8 s/step on Triton.

`ck.int8_linear` looks its implementation up on every call (`registry.get_implementation` inside
the function body), so re-enabling a backend here -- while custom nodes load, before any
inference -- is enough. Nothing else needs to know, and no launcher flag is required.

CUDA stays preferred when it is available: Triton is only enabled as the fallback when the
CUDA backend is disabled. `--disable-triton-backend`, or `LYNNREAL_NO_TRITON=1`, opts out.
"""

import logging
import os

import comfy.cli_args
import comfy.quant_ops


def _activate():
    if not getattr(comfy.quant_ops, "_CK_AVAILABLE", False):
        logging.warning("LynnReal: comfy_kitchen is unavailable, INT8 models will not load.")
        return

    ck = comfy.quant_ops.ck
    backends = ck.list_backends()
    cuda = backends.get("cuda") or {}
    triton = backends.get("triton") or {}

    if os.environ.get("LYNNREAL_NO_TRITON") == "1":
        logging.info("LynnReal: leaving the comfy-kitchen backends alone (LYNNREAL_NO_TRITON=1).")
        return
    if getattr(comfy.cli_args.args, "disable_triton_backend", False):
        logging.info("LynnReal: leaving the comfy-kitchen backends alone (--disable-triton-backend).")
        return
    if not cuda.get("disabled", True):
        return  # A cu130 torch: the CUDA backend is already the fast path.
    if not triton.get("available") or triton.get("disabled") is False:
        return

    ck.registry.enable("triton")
    logging.info(
        "LynnReal: enabled the comfy-kitchen Triton backend for INT8 models. The CUDA backend "
        "needs a cu130 torch and this environment does not have one, so without Triton the "
        "quantized GEMMs run eagerly (about 3x slower). Pass --enable-triton-backend to make "
        "this explicit, or --disable-triton-backend / LYNNREAL_NO_TRITON=1 to opt out."
    )


_activate()


from comfy_api.latest import ComfyExtension, io  # noqa: E402
from typing_extensions import override  # noqa: E402


class LynnRealInt8Backend(io.ComfyNode):
    """Report which comfy-kitchen backend will serve the quantized ops."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LynnRealInt8Backend",
            display_name="LynnReal INT8 backend info",
            description=(
                "Shows which comfy-kitchen backend serves the quantized LynnReal models. Triton is "
                "the fast fallback when the CUDA backend is unavailable; the eager backend is "
                "correct but about 3x slower."
            ),
            category="model/conditioning/minimax",
            inputs=[],
            outputs=[io.String.Output(display_name="backend")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        backends = comfy.quant_ops.ck.list_backends() if getattr(comfy.quant_ops, "_CK_AVAILABLE", False) else {}
        active = [name for name, info in backends.items() if info.get("available") and not info.get("disabled")]
        report = "comfy-kitchen backends active: {}".format(", ".join(active) or "none")
        if "triton" in active:
            report += "  |  INT8 GEMM: Triton (fast)"
        elif "cuda" in active:
            report += "  |  INT8 GEMM: CUDA (fast)"
        else:
            report += "  |  INT8 GEMM: eager (correct but ~3x slower)"
        return io.NodeOutput(report, ui={"text": (report,)})


class LynnRealBackendsExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LynnRealInt8Backend]


async def comfy_entrypoint() -> LynnRealBackendsExtension:
    return LynnRealBackendsExtension()
