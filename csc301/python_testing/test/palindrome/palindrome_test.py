import unittest
import logging
from palindrome.palindrome import is_palindrome

class TestPalindrome(unittest.TestCase):

    def test_none(self):
        self.assertFalse(is_palindrome(None))

    def test_palindrome(self):
        self.assertTrue(is_palindrome("abba"))

    def test_notpalindrome(self):
        self.assertFalse(is_palindrome("hello"))

    def test_empty(self):
        self.assertTrue(is_palindrome(""))

    def test_number(self):
        self.assertRaises(TypeError, is_palindrome, 99)


if __name__ == "__main__":
    unittest.main()