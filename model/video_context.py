"""A bounded, labelled view of actual video history for the native image encoder."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def motion_sheet(frames, appearance=None):
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2:
        raise ValueError("motion context requires at least two RGB video frames")
    height, width = frames.shape[1:3]
    if height % 2 or width % 2:
        raise ValueError("context canvas must have even dimensions")
    indices = np.linspace(0, len(frames)-1, 4).round().astype(int)
    images = [frames[i] for i in indices]
    labels = [f"HISTORY {i+1} / 4" for i in range(4)]
    if appearance is not None:
        if appearance.shape != frames[0].shape:
            raise ValueError("appearance and video canvases must agree")
        images = [appearance, frames[0], frames[len(frames)//2], frames[-1]]
        labels = ["INITIAL APPEARANCE", "EARLIER HISTORY", "RECENT HISTORY", "CURRENT BOUNDARY"]
    sheet = Image.new("RGB", (width, height))
    for i, (pixels, label) in enumerate(zip(images, labels)):
        x, y = i % 2 * (width//2), i // 2 * (height//2)
        tile = Image.fromarray(pixels).resize((width//2, height//2), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, width//2, 22), fill="black")
        draw.text((6, 4), label, fill="white", font=ImageFont.load_default())
        sheet.paste(tile, (x, y))
    return sheet
