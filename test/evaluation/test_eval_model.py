import os
from unittest import TestCase

from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.raw_information_source import CSVFile
from orange_cb_recsys.evaluation import Serendipity, Novelty, CatalogCoverage
from orange_cb_recsys.evaluation.classification_metrics import Precision, Recall, FNMeasure, MRR
from orange_cb_recsys.evaluation.eval_model import RankingAlgEvalModel, ReportEvalModel
from orange_cb_recsys.evaluation.partitioning import KFoldPartitioning
from orange_cb_recsys.evaluation.ranking_metrics import NDCG, Correlation
from orange_cb_recsys.recsys import CosineSimilarity, ClassifierRecommender
from orange_cb_recsys.recsys.ranking_algorithms.classifier import SVM
from orange_cb_recsys.recsys.config import RecSysConfig

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
contents_path = os.path.join(THIS_DIR, "../../contents")
datasets_path = os.path.join(THIS_DIR, "../../datasets")
ratings_filename = os.path.join(datasets_path, "examples/new_ratings.csv")
users_dir = os.path.join(contents_path, "examples/ex_1/users_1600355755.1935306")
items_dir = os.path.join(contents_path, "examples/ex_1/movies_1600355972.49884")

t_ratings = RatingsImporter(
    source=CSVFile(ratings_filename),
    rating_configs=[RatingsFieldConfig(
        field_name='points',
        processor=NumberNormalizer(min_=1, max_=5))],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
).import_ratings()

class TestRankingEvalModel(TestCase):
    def test_fit(self):

        recsys_config = RecSysConfig(
            users_directory=users_dir,
            items_directory=items_dir,
            score_prediction_algorithm=None,
            ranking_algorithm=ClassifierRecommender(
                {'Plot': '0'}, SVM()),
            rating_frame=t_ratings
        )
        print(t_ratings)
        ranking = RankingAlgEvalModel(
            config=recsys_config,
            partitioning=KFoldPartitioning(),
            metric_list=
            [
                Precision(0.4),
                Recall(0.4),
                FNMeasure(1, 0.4),
                MRR(0.4),
                NDCG({0: (-1, 0), 1: (0, 1)}),
                Correlation('pearson'),
                Correlation('kendall'),
                Correlation('spearman'),
            ])
        ranking_alg_eval = ranking.fit()
        self.assertEqual(len(ranking_alg_eval.columns), 9)
        self.assertEqual(len(ranking_alg_eval), 70)

class TestReportEvalModel(TestCase):
    def test_fit(self):

        recsys_config = RecSysConfig(
            users_directory=users_dir,
            items_directory=items_dir,
            score_prediction_algorithm=None,
            ranking_algorithm=ClassifierRecommender(
                {'Plot': '0'}, SVM(), 0
            ),
            rating_frame=t_ratings
        )

        report =ReportEvalModel(
            config=recsys_config,
            recs_number=3,
            metric_list=
            [
             Serendipity(num_of_recs=3),
             Novelty(num_of_recs=3),
             CatalogCoverage()
        ])
        report_eval = report.fit()
        self.assertEqual(len(report_eval), 3)
