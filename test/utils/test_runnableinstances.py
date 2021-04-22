from unittest import TestCase
from orange_cb_recsys.utils.runnable_instances import *


class Test(TestCase):
    def test_runnable_instances(self):
        show()

        get()

        add('test', 'test_test')

        remove('test')

        show()

        add('test2', 'test_cat', 'preprocessor')

        with self.assertRaises(ValueError):
            add('test2', 'test_cat', 'test_fail')
        number_of_instances = len(get().keys())
        add('test2', 'test_cat')
        self.assertEqual(len(get().keys()), number_of_instances)
        remove('test3')
        self.assertEqual(len(get().keys()), number_of_instances)
        remove('test2')
        show(True)

        x = get_cat('preprocessor')
        self.assertIn("nltk", x)
        self.assertNotIn("embedding", x)
