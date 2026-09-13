"""Give only the second image reference its own native encoding resolution."""
from PIL import Image, ImageOps
from .reference import SizedReferenceSetup
from diffusers.modular_pipelines.minimax_h3.packing_ref2va import prepare_reference_image, reference_media_to_uint8


def configure_boundary_reference(short_edge):
    if short_edge not in (128, 768):
        raise ValueError('boundary reference must use 128p or 768p')
    original = SizedReferenceSetup.__call__
    audit = {}

    def setup(self, components, state):
        components, state = original(self, components, state)
        block = self.get_block_state(state)
        images = [(entry, ref) for entry, ref in zip(block.references, state.get('prepared_references'))
                  if ref.kind == 'image']
        if len(images) != 2:
            raise RuntimeError('expected original first image and previous delivered boundary')
        entry, boundary = images[1]
        image = entry.image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(reference_media_to_uint8(image))
        image = ImageOps.exif_transpose(image).convert('RGB')
        scale = short_edge / min(image.size)
        width, height = (max(32, round(size*scale/32)*32) for size in image.size)
        boundary.image = prepare_reference_image(image, height, width)
        if images[0][1].image.size != (1344, 768):
            raise RuntimeError('original first-image resolution changed')
        audit.clear()
        audit.update(original_image_size=[1344, 768], boundary_image_size=[width, height],
                     original_image_rows=1008, boundary_image_rows=width*height//1024)
        return components, state

    SizedReferenceSetup.__call__ = setup
    return audit
