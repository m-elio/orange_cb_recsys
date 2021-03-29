from unittest import TestCase
from orange_cb_recsys.utils.string_cleaner import clean_with_unders, clean_no_unders


class TestStringCleaner(TestCase):
    def test_clean_with_unders(self):
        string = "ahdsf. a?"
        string_with_unders = clean_with_unders(string)
        self.assertEqual(string_with_unders, "ahdsf_a")

    def test_clean_no_unders(self):
        string = "ahd_..ip"
        string_no_unders = clean_no_unders(string)
        self.assertEqual(string_no_unders, "ahdip")