from unittest import TestCase
from orange_cb_recsys.utils.check_tokenization import check_not_tokenized, check_tokenized


class Test(TestCase):
    def test_check_tokenized(self):
        str_ = 'abcd efg'
        list_ = ['abcd', 'efg']
        x = check_tokenized(str_)
        self.assertEqual(x, ['abcd', 'efg'])
        y = check_tokenized(list_)
        self.assertEqual(y, ['abcd', 'efg'])
        z = check_not_tokenized(str_)
        self.assertEqual(z, 'abcd efg')
        s = check_not_tokenized(list_)
        self.assertEqual(s, 'abcd efg')