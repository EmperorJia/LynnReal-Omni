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

STATE = {"installed": False, "fallback_logged": False}


def selector_pairs(table):
    """{primary column: alternate column} for columns that carry a one-timestep variant.

    Conversion convention (see tools/build_adaln_hybrid.py): a column whose one-hot marker sits
    alone on row r is the alternate for the column whose marker sits alone on row r-1.  The full
    form's own time embedder gives a slightly different value for the same timestep when a forward
    carries a single timestep instead of several, which is why the t2v first step needs its own
    column; at runtime the batch shape says which one applies.
    """
    values = table.detach().float()
    lone = {}
    for column in range(values.shape[1]):
        rows = (values[:, column] == 1.0).nonzero().flatten().tolist()
        if len(rows) == 1:
            lone.setdefault(rows[0], []).append(column)
    pairs = {}
    for row, columns in lone.items():
        if len(columns) != 1:
            continue
        previous = lone.get(row - 1) or []
        if row > 0 and len(previous) == 1:
            pairs[previous[0]] = columns[0]
    return pairs


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
    diffusion = getattr(getattr(model_patcher, "model", model_patcher), "diffusion_model", None)
    if diffusion is None:
        return False
    table = getattr(diffusion, "adaln_t_table", None)
    if not looks_like_selector_table(table):
        return False
    dtype = _model_dtype(diffusion)
    pairs = selector_pairs(table)
    if pairs:
        logging.info("LynnReal: adaLN table carries %d one-timestep alternate column(s) %s",
                     len(pairs), {int(k): int(v) for k, v in pairs.items()})

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

        def forward(t_emb, _original=original, _dtype=dtype, _module=module, _pairs=pairs):
            coords = t_emb
            if _pairs and coords.shape[0] == 1:
                # a one-timestep forward takes the alternate column for the timesteps that have one
                chosen = coords.argmax(dim=-1)
                for row in range(coords.shape[0]):
                    alternate = _pairs.get(int(chosen[row]))
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
        STATE["installed"] = True
        logging.info("LynnReal: adaLN selector table detected -- %d projections run in %s so the "
                     "Lite table's timesteps match the full-form checkpoint bit for bit.",
                     patched, dtype)
    return bool(patched)
