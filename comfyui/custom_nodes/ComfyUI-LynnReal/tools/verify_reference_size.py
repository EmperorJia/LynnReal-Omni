#!/usr/bin/env python3
"""CPU-only regression check for official Ref2VA ``max`` reference sizing."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("lynnreal_reference_size_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stub_comfy_kitchen() -> None:
    """Avoid comfy-kitchen constructing a Triton CUDA driver on a CPU node."""
    module = types.ModuleType("comfy_kitchen")
    module.int8_attention_is_available = lambda: False

    def unavailable(*args, **kwargs):
        raise RuntimeError("comfy_kitchen is stubbed for this CPU-only sizing check")

    module.int8_attention = unavailable
    module.prequantize_int8_attention = unavailable
    module.int8_attention_from_prequantized = unavailable
    sys.modules["comfy_kitchen"] = module


class FakeImage:
    def __init__(self, height: int, width: int):
        self.shape = (1, height, width, 3)

    def __getitem__(self, key):
        return self


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", required=True)
    parser.add_argument("--pack", required=True)
    args = parser.parse_args()

    comfy_dir = os.path.abspath(args.comfy)
    sys.path.insert(0, comfy_dir)
    os.chdir(comfy_dir)
    sys.argv = [sys.argv[0], "--cpu"]
    stub_comfy_kitchen()
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy_extras import nodes_minimax_h3

    module = load_module(os.path.join(os.path.abspath(args.pack), "reference_size.py"))
    cases = {
        (1024, 1024): (2048, 2048),
        (1344, 768): (3584, 2048),
        (1000, 1501): (2048, 3072),
        (4096, 3072): (2720, 2048),
    }
    for source, expected in cases.items():
        actual = module.resolve_max_reference_size(*source)
        print("{}x{} -> {}x{}".format(*source, *actual))
        if actual != expected or actual[0] % 32 or actual[1] % 32:
            raise AssertionError("{} resolved to {}, expected {}".format(source, actual, expected))

    node = nodes_minimax_h3.MiniMaxH3ReferenceToVideo

    @classmethod
    def fake_execute(cls, **kwargs):
        image = kwargs["ref_images"]["ref_image_0"]
        return image.shape[2], image.shape[1]

    node.execute = fake_execute
    resize_calls = []

    def fake_resize(image, width, height, crop):
        resize_calls.append((width, height, crop))
        return FakeImage(height, width)

    nodes_minimax_h3._resize = fake_resize
    if not module.install():
        raise AssertionError("patch did not install")
    actual = node.execute(
        clip=None,
        prompt="",
        width=1344,
        height=768,
        length=22,
        ref_image_size="max",
        ref_images={"ref_image_0": FakeImage(768, 1344)},
    )
    if actual != (3584, 2048) or resize_calls != [(3584, 2048, "disabled")]:
        raise AssertionError("execute wrapper did not upscale through the official path: {} {}"
                             .format(actual, resize_calls))

    schema = node.define_schema()
    size_input = next(value for value in schema.inputs if value.id == "ref_image_size")
    image_input = next(value for value in schema.inputs if value.id == "ref_images")
    if "upscaling included" not in size_input.tooltip:
        raise AssertionError("ref_image_size tooltip was not updated")
    if "upscaling included" not in image_input.template.input.tooltip:
        raise AssertionError("reference image tooltip was not updated")

    print("PASS: max upscales to a 2048px short edge, aligns both axes to 32, and updates tooltips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
