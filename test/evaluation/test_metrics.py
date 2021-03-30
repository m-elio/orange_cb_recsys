from unittest import TestCase
import pathlib as pl
from orange_cb_recsys.evaluation import Precision, Recall, FNMeasure, NDCG, MRR, Correlation, GiniIndex, \
    PopRecsCorrelation, LongTailDistr, CatalogCoverage, PopRatioVsRecs, DeltaGap, Serendipity, Novelty
import pandas as pd

from orange_cb_recsys.evaluation.prediction_metrics import RMSE, MAE

score_frame_fairness = pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
                                               'to_id': ["aaa", "bbb", "aaa", "bbb", "ccc", "aaa", "ddd", "bbb"],
                                               'rating': [1.0, 0.5, 0.0, 0.5, 0.6, 0.2, 0.7, 0.8]})
truth_frame_fairness = pd.DataFrame.from_dict({'from_id': ["001", "001", "002", "002", "002", "003", "004", "004"],
                                               'to_id': ["aaa", "bbb", "aaa", "ddd", "ccc", "ccc", "ddd", "ccc"],
                                               'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})


class Test(TestCase):
    def test_perform_ranking_metrics(self):
        col = ["to_id", 'rating']
        truth_rank = {
            "item0": 1.0,
            "item1": 1.0,
            "item2": 0.85,
            "item3": 0.8,
            "item4": 0.7,
            "item5": 0.65,
            "item6": 0.4,
            "item7": 0.35,
            "item8": 0.2,
            "item9": 0.2
        }

        predicted_rank = {
            "item2": 0.9,
            "item5": 0.85,
            "item9": 0.75,
            "item0": 0.7,
            "item4": 0.65,
            "item1": 0.5,
            "item8": 0.2,
            "item7": 0.2,
        }

        truth_rank = pd.DataFrame(truth_rank.items(), columns=col)
        predicted_rank = pd.DataFrame(predicted_rank.items(), columns=col)

        results = {
            "Precision": Precision(0.75).perform(predicted_rank, truth_rank),
            "Recall": Recall(0.75).perform(predicted_rank, truth_rank),
            "F1": FNMeasure(1, 0.75).perform(predicted_rank, truth_rank),
            "F2": FNMeasure(2, 0.75).perform(predicted_rank, truth_rank),
            "NDCG":
                NDCG({0: (-1.0, 0.0), 1: (0.0, 0.3), 2: (0.3, 0.7), 3: (0.7, 1.0)}).perform(predicted_rank, truth_rank),
            "MRR": MRR(0.75).perform(predicted_rank, truth_rank),
            "pearson": Correlation('pearson').perform(predicted_rank, truth_rank),
            "kendall": Correlation('kendall').perform(predicted_rank, truth_rank),
            "spearman": Correlation('spearman').perform(predicted_rank, truth_rank)
        }

        real_results = {
            "Precision": 0.5,
            "Recall": 0.5,
            "F1": 0.5,
            "F2": 0.5,
            "NDCG": 0.908,
            "MRR": 0.8958333333333333,
            "pearson": 0.26,
            "kendall": 0.14,
            "spearman": 0.19,
        }
        for x in results.keys():
            a = round((int(results[x])), 2)
            results[x] = a

        for x in real_results.keys():
            a = round((int(real_results[x])), 2)
            real_results[x] = a

        self.assertEqual(results, real_results)

    def test_NDCG(self):
        score_frame = pd.DataFrame.from_dict({'to_id': ["bbb", "eee", "aaa", "ddd", "ccc", "fff", "hhh", "ggg"],
                                              'rating': [1.0, 1.0, 0.5, 0.5, 0.3, 0.3, 0.7, 0.8]})
        truth_frame = pd.DataFrame.from_dict({'to_id': ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh"],
                                              'rating': [0.8, 0.7, -0.4, 1.0, 0.4, 0.1, -0.3, 0.7]})

        ndcg_test1 = NDCG().perform(predictions=score_frame, truth=truth_frame)
        self.assertTrue(0.0 <= ndcg_test1 <= 1.0)

        ndcg_test2 = NDCG({}).perform(predictions=score_frame, truth=truth_frame)
        self.assertTrue(0.0 <= ndcg_test2 <= 1.0)

        split_dict = {1: (-0.5, 0.0), 2: (0.0, 0.5), 3: (0.5, 1.0)}
        ndcg_test3 = NDCG(split_dict).perform(predictions=score_frame, truth=truth_frame)
        self.assertTrue(0.0 <= ndcg_test3 <= 1.0)

        split_dict = {0: (-1.0, -0.5), 1: (-0.5, 0.0), 2: (0.0, 0.5), 3: (0.5, 1.0)}
        ndcg_test4 = NDCG(split_dict).perform(predictions=score_frame, truth=truth_frame)
        self.assertTrue(0.0 <= ndcg_test4 <= 1.0)

    def test_perform_fairness_metrics(self):
        gini_index = GiniIndex().perform(score_frame_fairness)
        self.assertEqual(round(gini_index["gini-index"][0], 3), 0.167)
        self.assertEqual(round(gini_index["gini-index"][1], 3), 0.364)
        self.assertEqual(round(gini_index["gini-index"][2], 3), 0.000)
        self.assertEqual(round(gini_index["gini-index"][3], 3), 0.033)
        PopRecsCorrelation('test', '.').perform(score_frame_fairness, truth_frame_fairness)
        path_pop_recs = pl.Path("./pop-recs_test.svg")
        self.assertEqual((str(path_pop_recs), path_pop_recs.is_file()), (str(path_pop_recs), True))

        LongTailDistr('test', '.').perform(score_frame_fairness, truth_frame_fairness)
        path_long_tail_distr = pl.Path("./recs-long-tail-distr_test.svg")
        self.assertEqual((str(path_long_tail_distr), path_long_tail_distr.is_file()), (str(path_long_tail_distr), True))

        catalog_coverage = CatalogCoverage().perform(score_frame_fairness, truth_frame_fairness)
        self.assertEqual(catalog_coverage, 100)

        pop_ratio_vs_recs = PopRatioVsRecs('test', '.', {'niche': 0.2, 'diverse': 0.6, 'bb_focused': 0.2}, False).\
            perform(score_frame_fairness, truth_frame_fairness)
        path_pop_ratio_vs_recs = pl.Path("./pop_ratio_profile_vs_recs_test.svg")
        self.assertEqual((str(path_pop_ratio_vs_recs), path_pop_ratio_vs_recs.is_file()), (str(path_pop_ratio_vs_recs),
                                                                                           True))
        list_profile = pop_ratio_vs_recs["profile_pop_ratio"].to_list()
        value_list = [item for sublist in list_profile for item in sublist]
        for v in value_list:
            self.assertTrue(0.0 <= v <= 1.0)
        list_recs = pop_ratio_vs_recs["recs_pop_ratio"].to_list()
        value_list_recs = [item for sublist in list_recs for item in sublist]
        for v in value_list_recs:
            self.assertTrue(0.0 <= v <= 1.0)

        delta = DeltaGap({'niche': 0.2, 'diverse': 0.6, 'bb_focused': 0.2}).perform(score_frame_fairness,
                                                                                    truth_frame_fairness)
        list_delta = delta["delta-gap"].to_list()
        self.assertEqual(len(list_delta), 3)
        print(delta)

    def test_perform_serendipity(self):
        true_serendipity = 0.175
        serendipity = Serendipity(10).perform(score_frame_fairness, truth_frame_fairness)
        self.assertEqual(round(serendipity, 3), true_serendipity)

    def test_perform_novelty(self):
        true_novelty = 0.3165
        novelty = Novelty(10).perform(score_frame_fairness, truth_frame_fairness)
        novelty = round(novelty, 4)
        self.assertEqual(novelty, true_novelty)

    def test_perform_rmse(self):
        predictions = pd.DataFrame.from_dict({'to_id': ["bbb", "eee", "aaa", "ddd", "ccc", "fff", "hhh"],
                                              'rating': [5, 5, 4, 3, 3, 2, 1]})
        truth = pd.DataFrame.from_dict({'to_id': ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg"],
                                        'rating': [5, 4, 3, 3, 1, 2, 1]})

        self.assertEqual(RMSE().perform(predictions, truth), 0.9258200997725514)
        self.assertEqual(MAE().perform(predictions, truth), 0.5714285714285714)

        truth_exception = pd.DataFrame.from_dict({'to_id': ["aaa", "bbb", "ccc", "ddd", "eee", "fff"],
                                                  'scores': [5, 4, 3, 3, 1, 2]})

        with self.assertRaises(Exception):
            RMSE().perform(predictions, truth_exception)

        less_predictions = predictions[:-1]
        self.assertEqual(round(RMSE().perform(less_predictions, truth)), 1.0)
        self.assertEqual(round(MAE().perform(less_predictions, truth), 3), 0.667)

        less_truth = truth[:-1]
        self.assertEqual(round(RMSE().perform(predictions, less_truth)), 1.0)
        self.assertEqual(round(MAE().perform(predictions, less_truth), 3), 0.667)
