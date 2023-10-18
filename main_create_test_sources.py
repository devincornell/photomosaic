from __future__ import annotations
import pathlib
import dataclasses
import typing
from PIL import Image
import numpy as np
import random
import canvas

if __name__ == "__main__":
    cv = canvas.Canvas.from_image(pathlib.Path("data/test/black_circle2.png"))
    
    subcanvases = cv.split_subcanvases(3,3)
    for i in range(100):
        for i, sc in enumerate(subcanvases):
            y = random.randrange(0,255)
            sc.x += y            
            sc.image().save(f"data/test_original/rand-{y}.png")

