from unittest import TestCase
from orange_cb_recsys.evaluation.delta_gap import calculate_gap, calculate_delta_gap


class Test(TestCase):
    def test_calculate_gap(self):
        group = {'aaa', 'bbb', 'ccc'}
        avg_pop = {'aaa': 0.5, 'bbb': 0.7}
        gap = calculate_gap(group, avg_pop)
        self.assertEqual(round(gap, ndigits=2), 0.4)

    def test_calculate_delta_gap(self):
        delta_gap = calculate_delta_gap(1.0, 0.0)
        self.assertEqual(delta_gap, 0.0)

