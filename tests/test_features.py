import unittest

import numpy as np
from PIL import Image

from listing_verifier.features.dct import dct_matrix, log_dct_2d, radial_profile, spectrum_features
from listing_verifier.features.image_ops import canonicalise, degrade, jpeg_bytes_size, to_square_gray


class DctTests(unittest.TestCase):
    def test_orthonormal(self):
        D = dct_matrix(16)
        self.assertTrue(np.allclose(D @ D.T, np.eye(16), atol=1e-5))

    def test_constant_image_energy_in_dc(self):
        spec = log_dct_2d(np.full((32, 32), 0.5, np.float32))
        self.assertTrue(spec[0, 0] > spec[1:, 1:].max() + 5)

    def test_shapes(self):
        f = spectrum_features(np.random.rand(256, 256).astype(np.float32))
        self.assertEqual(f["dct_map"].shape, (64, 64))
        self.assertEqual(f["dct_radial"].shape, (128,))
        self.assertEqual(radial_profile(np.zeros((256, 256), np.float32)).shape, (128,))

    def test_jpeg_removes_high_frequencies(self):
        rng = np.random.default_rng(0)
        img = Image.fromarray((rng.random((256, 256, 3)) * 255).astype("uint8"))
        hi = radial_profile(log_dct_2d(to_square_gray(canonicalise(img))))[-16:].mean()
        lo = radial_profile(log_dct_2d(to_square_gray(degrade(canonicalise(img), 30))))[-16:].mean()
        self.assertLess(lo, hi)  # the frequency cue really is erased by compression


class ImageOpsTests(unittest.TestCase):
    def test_canonical_size(self):
        img = Image.new("RGB", (1024, 512), (200, 10, 10))
        c = canonicalise(img)
        self.assertEqual(max(c.size), 256)
        self.assertEqual(canonicalise(Image.new("RGB", (100, 80))).size, (100, 80))

    def test_degrade_smaller_file(self):
        img = Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8"))
        self.assertLess(jpeg_bytes_size(img, 30), jpeg_bytes_size(img, 90))


if __name__ == "__main__":
    unittest.main()
