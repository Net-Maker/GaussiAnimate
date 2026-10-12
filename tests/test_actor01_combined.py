import importlib.util
from pathlib import Path
import unittest

import numpy as np
from PIL import Image


class CombinedPreviewTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/compose_actor01_comparison.py'
        spec = importlib.util.spec_from_file_location('combined', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_error_is_zero_for_identical_images(self):
        image = Image.new('RGB', (40, 40), (10, 30, 50))
        self.assertEqual(np.asarray(self.module.error_image(image, image)).max(), 0)

    def test_six_panels_keep_order(self):
        colors = [(80 + i * 20, 10, 20) for i in range(6)]
        images = [Image.new('RGB', (20, 20), color) for color in colors]
        canvas, boxes = self.module.compose(images, 240, 120)
        self.assertEqual(len(boxes), 6)
        for box, color in zip(boxes, colors):
            x, y, w, h = box
            self.assertEqual(canvas.getpixel((x + w // 2, y + h // 2)), color)


if __name__ == '__main__':
    unittest.main()
