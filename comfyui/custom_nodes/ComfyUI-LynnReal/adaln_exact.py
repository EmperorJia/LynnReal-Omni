"""Run a one-hot adaLN table with the same arithmetic the full-form checkpoint uses.

The Flash checkpoints only ever ask the adaLN projection for a handful of timesteps, so a table can
carry the *literal* modulation vectors the full form produced for them, selected by a one-hot row
(see `tools/build_adaln_hybrid.py`; the same table also keeps a rank-8 curve in its other columns so
any other schedule degrades to the released curve quality instead of misbehaving).

ComfyUI builds the curve path with `adaln_dtype=torch.float32`, which is the right choice for a
continuous curve but not for a lookup: the full form computes the modulation in the model dtype
(bf16), and this sampler amplifies a bf16 ulp into a different clip.  So when a loaded model's table
is a selector table, this module casts that model's adaLN linears to the model dtype and casts the
interpolated coordinates before the projection, which makes the visited timesteps bit-identical to
the full form.

Detection is structural, so users need no flag and nothing under `comfy/` is patched on disk:
`install()` is called from the token-compression node, which every Flash workflow already uses.
"""

from __future__ import annotations

import logging

import torch

STATE = {"installed": False, "fallback_logged": False, "global_hook": False}


def selector_variants(table):
    """{(primary column, batch rows): alternate column} for GEMM-shape variants.

    Conversion convention (see tools/build_adaln_hybrid.py): a primary selector occupies rows
    ``r,r+1``.  A batch-M variant has a lone marker at ``r+M``.  The t=0/M=1 special case uses a
    lone primary at row 0 and its alternate at row 1.  The full form's BF16 GEMM can round
    differently for M=1, M=2, and M=3, so Standard t2v needs shape-specific columns at every step.
    """
    values = table.detach().float()
    rows_by_column = {}
    for column in range(values.shape[1]):
        rows = (values[:, column] == 1.0).nonzero().flatten().tolist()
        if rows:
            rows_by_column[column] = rows
    primaries = {}
    for column, rows in rows_by_column.items():
        if len(rows) == 2 and rows[1] == rows[0] + 1:
            primaries[column] = rows[0]
        elif rows == [0]:
            primaries[column] = 0
    lone_at = {}
    for column, rows in rows_by_column.items():
        if len(rows) == 1:
            lone_at.setdefault(rows[0], []).append(column)
    variants = {}
    for primary, base in primaries.items():
        for batch_rows in range(1, 9):
            alternates = [column for column in lone_at.get(base + batch_rows, [])
                          if column != primary]
            if len(alternates) == 1:
                variants[(primary, batch_rows)] = alternates[0]
    return variants


def looks_like_selector_table(table) -> bool:
    """True for a lookup table (has one-hot rows), False for a continuous curve table."""
    if table is None or table.ndim != 2 or table.shape[1] < 2:
        return False
    values = table.detach().float()
    one_hot = (values.max(dim=1).values == 1.0) & (values.sum(dim=1) == 1.0)
    return bool(one_hot.any()) and int(one_hot.sum()) <= 64


def _model_dtype(diffusion):
    for name in ("condition_proj", "audio_patch_proj", "video_patch_proj"):
        module = getattr(diffusion, name, None)
        if module is not None and getattr(module, "weight", None) is not None:
            return module.weight.dtype
    return torch.bfloat16


def install(model_patcher) -> bool:
    """Patch this model's adaLN projections if its table is a selector table."""
    root = getattr(model_patcher, "model", model_patcher)
    diffusion = getattr(root, "diffusion_model", None)
    if diffusion is None and type(root).__name__ == "MiniMaxH3Model":
        diffusion = root
    if diffusion is None:
        return False
    if getattr(diffusion, "_lynnreal_exact_installed", False):
        return True
    table = getattr(diffusion, "adaln_t_table", None)
    if not looks_like_selector_table(table):
        return False
    dtype = _model_dtype(diffusion)
    variants = selector_variants(table)
    if variants:
        logging.info("LynnReal: adaLN table carries %d batch-shape variant column(s) %s",
                     len(variants), {"%d@M%d" % (int(k[0]), int(k[1])): int(v)
                                     for k, v in variants.items()})

    patched = 0
    for module in diffusion.modules():
        if type(module).__name__ != "AdalnProj":
            continue
        linear = module.linear
        if getattr(linear, "_lynnreal_exact", False):
            continue
        with torch.no_grad():
            linear.weight.data = linear.weight.data.to(dtype)
            if linear.bias is not None:
                linear.bias.data = linear.bias.data.to(dtype)
        original = module.forward

        def forward(t_emb, _original=original, _dtype=dtype, _module=module,
                    _variants=variants):
            coords = t_emb
            if _variants:
                chosen = coords.argmax(dim=-1)
                for row in range(coords.shape[0]):
                    alternate = _variants.get((int(chosen[row]), coords.shape[0]))
                    if alternate is None:
                        continue
                    coords = coords.clone()
                    coords[row] = 0.0
                    coords[row, alternate] = 1.0
            with torch.no_grad():
                is_one_hot = bool(((coords.max(dim=-1).values == 1.0)
                                   & (coords.sum(dim=-1) == 1.0)).all())
            if not is_one_hot and not STATE["fallback_logged"]:
                STATE["fallback_logged"] = True
                logging.info("LynnReal: adaLN timesteps outside the Lite table; using the curve "
                             "columns of the hybrid table for this run.")
            return _original(coords.to(_dtype))

        module.forward = forward
        linear._lynnreal_exact = True
        patched += 1

    if patched:
        diffusion._lynnreal_exact_installed = True
        STATE["installed"] = True
        logging.info("LynnReal: adaLN selector table detected -- %d projections run in %s so the "
                     "Lite table's timesteps match the full-form checkpoint bit for bit.",
                     patched, dtype)
    return bool(patched)


def install_global() -> None:
    """Install selector handling for Standard workflows that have no Flash compression node."""
    if STATE["global_hook"]:
        return
    from comfy.ldm.minimax.model import MiniMaxH3Model

    original = MiniMaxH3Model.forward

    def forward(self, *args, **kwargs):
        install(self)
        return original(self, *args, **kwargs)

    MiniMaxH3Model.forward = forward
    STATE["global_hook"] = True
