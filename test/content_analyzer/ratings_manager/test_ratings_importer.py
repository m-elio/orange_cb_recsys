import glob
import os
from unittest import TestCase
import pathlib as pl
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsImporter, RatingsFieldConfig
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(THIS_DIR, "../../../datasets/test_import_ratings.json")
file_path2 = os.path.join(THIS_DIR, "test_ratings_*.csv")


class TestRatingsImporter(TestCase):
    def test_import_ratings(self):
        RatingsImporter(source=JSONFile(file_path=file_path),
                        output_directory="test_ratings",
                        rating_configs=[
                            RatingsFieldConfig(field_name="review_title",
                                               processor=TextBlobSentimentAnalysis()),
                            RatingsFieldConfig(field_name="text",
                                               processor=TextBlobSentimentAnalysis()),
                            RatingsFieldConfig(field_name="stars",
                                               processor=NumberNormalizer(min_=0, max_=5))],
                        from_field_name="user_id",
                        to_field_name="item_id",
                        timestamp_field_name="timestamp").import_ratings()
        x = sorted(glob.glob(file_path2))[-1]
        dynamic_path = pl.Path(x)
        self.assertEqual((str(dynamic_path), dynamic_path.is_file()), (str(dynamic_path), True))