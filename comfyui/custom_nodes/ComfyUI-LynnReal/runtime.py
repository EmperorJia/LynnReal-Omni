"""Memory policy for the LynnReal graphs, without asking the user for launcher flags.

The standard workflows load a 61.7 GiB bf16 DiT plus a 15 GiB text encoder on an 80 GB card.
With ComfyUI's default (a few hundred MB aside) both stay resident and every run costs about
10 s of overhead on top of the sampler; measured warm standard t2v: 46-51 s.

Raising the reserve buys headroom at a large price: ComfyUI then evicts the 15 GiB text encoder
after every run and reloads it on the next one (plus a partial DiT unload), which added ~38 s
per run in our measurements (46 s -> 85 s for the same 4-step 1344x768 t2v, with identical
8.5-9.2 s/it sampling).

So the pack keeps ComfyUI's own reserve. Two things that help on this hardware are set up here:

* **DynamicVRAM on a cu13x torch.** ComfyUI enables it automatically when the torch build is
  >= 2.8 and CUDA >= 13 (main.py does this before custom nodes load), and it manages the
  resident/offloaded split itself. If the detection says otherwise, the pack tries the same
  initialisation main.py uses and reports what happened.
* **A default of no extra reserve** on the older stacks (cu126/torch 2.7 in the release
  environment), because that is what keeps the DiT and the text encoder resident.

The pose control workflow is the exception: a frame-aligned control clip packs reference and
target into a single sequence, and on a card where the DiT is fully resident that dies with

    torch.OutOfMemoryError: Allocation on device
    Allocated 68.6 GiB, peak 76.2 GiB, reserved 80.3 GiB

Run that workflow with ``--reserve-vram 10`` (or ``LYNNREAL_RESERVE_VRAM=10``); it costs ~30 s
per run because of the text-encoder eviction, and it is only needed for the control graphs.

Overrides:

* ``--reserve-vram`` on the command line always wins (any value, including 0);
* ``LYNNREAL_RESERVE_VRAM=<GB>`` sets the reserve for the whole process (``0`` keeps ComfyUI's
  default);
* ``LYNNREAL_NO_DYNAMIC_VRAM=1`` leaves the DynamicVRAM decision to ComfyUI alone.
"""

import logging
import os

import comfy.cli_args
import comfy.model_management

GB = 1024 ** 3


def _fixed_reserve():
    """The reserve the user asked for, or None when they did not."""
    if comfy.cli_args.args.reserve_vram is not None:
        return float(comfy.cli_args.args.reserve_vram)
    raw = os.environ.get("LYNNREAL_RESERVE_VRAM")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        logging.warning("LynnReal: LYNNREAL_RESERVE_VRAM=%r is not a number; ignoring it.", raw)
        return None


def _cuda_major() -> int | None:
    raw = getattr(__import__("torch").version, "cuda", None)
    if not raw:
        return None
    try:
        return int(str(raw).split(".")[0])
    except ValueError:
        return None


def _dynamic_vram() -> None:
    """Report DynamicVRAM, and turn it on from here if ComfyUI could not (cu13x torch)."""
    import comfy.memory_management as memory_management

    if getattr(memory_management, "aimdo_enabled", False):
        logging.info("LynnReal: DynamicVRAM is active; ComfyUI manages the resident/offloaded "
                     "split itself.")
        return
    if os.environ.get("LYNNREAL_NO_DYNAMIC_VRAM") == "1":
        logging.info("LynnReal: leaving the DynamicVRAM decision to ComfyUI "
                     "(LYNNREAL_NO_DYNAMIC_VRAM=1).")
        return

    cuda_major = _cuda_major()
    if cuda_major is None or cuda_major < 13:
        logging.info(
            "LynnReal: torch is CUDA %s; DynamicVRAM needs a cu13x torch (and torch >= 2.8), so "
            "ComfyUI is on its legacy model loader. The pack therefore keeps ComfyUI's own VRAM "
            "reserve -- full-speed for t2v/i2v/r2v/v2v. Run the pose control workflow with "
            "--reserve-vram 10 (it needs the headroom and pays ~30 s/run for it).", cuda_major)
        return

    try:
        import torch
        version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
        if version < (2, 8):
            logging.warning("LynnReal: cu13x detected but torch %s is older than 2.8; DynamicVRAM "
                            "is unsupported on this build.", torch.__version__)
            return
        import comfy.model_management as model_management
        import comfy.model_patcher as model_patcher
        import comfy_aimdo.control as aimdo_control

        headroom = int(comfy.cli_args.args.vram_headroom * GB)
        devices = list(model_management.get_all_torch_devices())
        try:
            initialized = aimdo_control.init_devices((device.index, headroom) for device in devices)
        except TypeError:  # comfy-aimdo 0.4.9 protocol
            initialized = aimdo_control.init_devices(device.index for device in devices)
        if initialized:
            model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
            memory_management.aimdo_enabled = True
            logging.info("LynnReal: DynamicVRAM enabled by the pack (cu13x torch, %d device(s)).",
                         len(devices))
        else:
            logging.warning("LynnReal: comfy-aimdo reported no devices; launch ComfyUI with "
                            "--enable-dynamic-vram to see its own diagnostics.")
    except Exception as error:  # pragma: no cover - depends on the installed comfy-aimdo
        logging.warning("LynnReal: could not enable DynamicVRAM (%s); launch ComfyUI with "
                        "--enable-dynamic-vram if you want it.", error)


def _activate() -> None:
    _dynamic_vram()
    fixed = _fixed_reserve()
    if fixed is None:
        logging.info(
            "LynnReal: keeping ComfyUI's default VRAM reserve (fast path). Graphs that need "
            "headroom, such as the pose control workflow, take --reserve-vram 10.")
        return
    if fixed <= 0:
        logging.info("LynnReal: ComfyUI's default VRAM reserve kept (reserve pinned to 0).")
        return
    comfy.model_management.EXTRA_RESERVED_VRAM = int(fixed * GB)
    logging.info("LynnReal: reserving %.0f GB of VRAM for the whole process, as requested.", fixed)


_activate()
