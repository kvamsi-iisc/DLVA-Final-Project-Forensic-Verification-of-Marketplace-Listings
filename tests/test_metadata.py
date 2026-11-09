import unittest

from listing_verifier.data.abo_metadata import (extract_count, family_key, is_placeholder_description,
                                                normalise_colour, normalise_material, product_noun)


class MetadataTests(unittest.TestCase):
    def test_colour_single(self):
        self.assertEqual(normalise_colour("Navy Blue"), "blue")
        self.assertEqual(normalise_colour("Espresso"), "brown")
        self.assertEqual(normalise_colour("Off-White"), "white")

    def test_colour_ambiguous_is_none(self):
        self.assertIsNone(normalise_colour("Black/White"))
        self.assertIsNone(normalise_colour("Multicolor Yellow Accent"))
        self.assertIsNone(normalise_colour("Teal"))
        self.assertIsNone(normalise_colour(None))

    def test_colour_standardized_values_used(self):
        self.assertEqual(normalise_colour("Midnight", ["Black"]), "black")

    def test_material(self):
        self.assertEqual(normalise_material("Faux Leather"), "leather")
        self.assertEqual(normalise_material("100% Cotton"), "fabric")
        self.assertIsNone(normalise_material("Wood and Metal"))

    def test_count(self):
        self.assertEqual(extract_count("Set of 4 Dining Chairs"), (4, "regex"))
        self.assertEqual(extract_count("Bath Towel 2-Pack"), (2, "regex"))
        self.assertEqual(extract_count("Rivet Sofa"), (1, "inferred_single"))
        self.assertEqual(extract_count("Cutlery Set"), (None, None))
        self.assertEqual(extract_count("Pack of 48 batteries"), (None, None))

    def test_placeholder(self):
        self.assertTrue(is_placeholder_description("Content coming soon"))
        self.assertTrue(is_placeholder_description(""))
        self.assertFalse(is_placeholder_description("A sturdy oak table with four legs and a drawer."))

    def test_family_key_groups_variants(self):
        a = family_key("Rivet", 'Rivet Modern Leather Sofa, 80"W, Black')
        b = family_key("Rivet", 'Rivet Modern Leather Sofa, 80"W, Brown')
        c = family_key("Rivet", "Rivet Round Side Table, Walnut")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_product_noun(self):
        self.assertEqual(product_noun("CELLULAR_PHONE_CASE"), "cellular phone case")
        self.assertEqual(product_noun(None), "item")


if __name__ == "__main__":
    unittest.main()
