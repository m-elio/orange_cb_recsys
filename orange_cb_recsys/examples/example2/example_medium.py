from orange_cb_recsys.content_analyzer import ContentAnalyzer, ContentAnalyzerConfig
from orange_cb_recsys.content_analyzer.config import FieldConfig, FieldRepresentationPipeline
from orange_cb_recsys.content_analyzer.field_content_production_techniques import SearchIndexing, LuceneTfIdf, \
    BabelPyEntityLinking, SynsetDocumentFrequency
from orange_cb_recsys.content_analyzer.information_processor import NLTK
from orange_cb_recsys.content_analyzer.ratings_manager import RatingsImporter
from orange_cb_recsys.content_analyzer.ratings_manager.rating_processor import NumberNormalizer
from orange_cb_recsys.content_analyzer.ratings_manager.ratings_importer import RatingsFieldConfig
from orange_cb_recsys.content_analyzer.ratings_manager.sentiment_analysis import TextBlobSentimentAnalysis
from orange_cb_recsys.content_analyzer.raw_information_source import JSONFile
from orange_cb_recsys.recsys import ClassifierRecommender
from orange_cb_recsys.recsys.recsys import RecSys, RecSysConfig

movies_filename = '../../../datasets/movies_info_reduced.json'
user_filename = '../../../datasets/users_info_.json'
ratings_filename = '../../../datasets/ratings_example.json'

output_dir = '../../../contents/test_1m_medium/dir'
"""
movies_ca_config = ContentAnalyzerConfig(
    content_type='Item',
    source=JSONFile(movies_filename),
    id_field_name_list=['imdbID'],
    output_directory=output_dir,
)

movies_ca_config.append_field_config(
    field_name='Plot',
    field_config=FieldConfig(
        pipelines_list=[
            FieldRepresentationPipeline(preprocessor_list=[NLTK(lemmatization=True)], 
            content_technique=SynsetDocumentFrequency())]
    )
)

movies_ca_config.append_field_config(
    field_name='Actors',
    field_config=FieldConfig(
        pipelines_list=[
            FieldRepresentationPipeline(content_technique=SynsetDocumentFrequency())]
    )
)

content_analyzer_movies = ContentAnalyzer(
    config=movies_ca_config
)

user_ca_config = ContentAnalyzerConfig(
    content_type='User',
    source=JSONFile(user_filename),
    id_field_name_list=["user_id"],
    output_directory=output_dir
)

user_ca_config.append_field_config(
    field_name="name",
    field_config=FieldConfig(
        pipelines_list=[FieldRepresentationPipeline(
            preprocessor_list=[NLTK(url_tagging=True, strip_multiple_whitespaces=True)], content_technique=None)]
    )
)

content_analyzer_users = ContentAnalyzer(
    config=user_ca_config
)

content_analyzer_movies.fit()
content_analyzer_users.fit()
"""
title_review_config = RatingsFieldConfig(
    field_name='review_title',
    processor=TextBlobSentimentAnalysis()
)

starts_review_config = RatingsFieldConfig(
    field_name='stars',
    processor=NumberNormalizer(min_=1, max_=5))

ratings_importer = RatingsImporter(
    source=JSONFile(ratings_filename),
    rating_configs=[title_review_config, starts_review_config],
    from_field_name='user_id',
    to_field_name='item_id',
    timestamp_field_name='timestamp',
)

ratings_frame = ratings_importer.import_ratings()

classifier_config = ClassifierRecommender(
    item_field='Plot',
    field_representation='0',
    classifier='random_forest'
)

classifier_recsys_config = RecSysConfig(
    users_directory='../../../contents/test_1m_medium/dir1600079012.1683488',
    items_directory='../../../contents/test_1m_medium/dir1600079332.8693523',
    ranking_algorithm=classifier_config,
    rating_frame=ratings_frame
)

classifier_recommender = RecSys(
    config=classifier_recsys_config
)

rank =classifier_recommender.fit_ranking(
    user_id='01',
    recs_number=10
)
print(rank)