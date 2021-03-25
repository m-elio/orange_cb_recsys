from typing import List

from orange_cb_recsys.content_analyzer.content_representation.content_field import EmbeddingField, FeaturesBagField
from orange_cb_recsys.recsys.algorithm import ItemFieldRankingAlgorithm, Transformer
from orange_cb_recsys.recsys.ranking_algorithms.similarities import Similarity, DenseVector, CosineSimilarity
import pandas as pd
import numpy as np

from orange_cb_recsys.utils.const import logger


class CentroidVector(Transformer):
    """
    Class used to compute the centroid of positive rated items by a user and
    the similarity between the centroid and the items not rated by said user.

    Args:
        similarity (Similarity): similarity implementation to compute
    """

    def __init__(self, similarity: Similarity):
        super().__init__()
        self.__centroid = None
        self.__similarity = similarity

    def compute_centroid(self, positive_rated_items: list):
        """
        Transforms the positive rated item fields representations into a single one, using the
        method transform of the Transformer class, and computes the centroid of the items positively rated by a user.

        Args:
            positive_rated_items (list): list of positive item representations for each field
        """
        positive_rated_items = super().transform(positive_rated_items)

        self.__centroid = np.array(positive_rated_items).mean(axis=0)

    def compute_similarities(self, unrated_items: list):
        """
        Transforms the unrated item fields representations, using the method transform of the Transformer class, and
        computes the similarity between the centroid and the unrated items.

        Args:
            unrated_items (list): list of unrated item representations for each field

        Returns:
            similarity (list): list of similarity scores between each unrated item and the positive rated items centroid
        """
        unrated_items = super().transform(unrated_items)

        dense_centroid = DenseVector(self.__centroid)
        similarity = [self.__similarity.perform(dense_centroid, DenseVector(item)) for item in unrated_items]

        return similarity


class CentroidVectorRecommender(ItemFieldRankingAlgorithm):
    """
    Class that implements a centroid-like recommender. It first gets the centroid of the items that the user liked and
    then computes the similarity between the centroid and the item of which predict the score, using a CentroidVector
    object instantiated inside the class. The similarity can be chosen and passed as an argument, otherwise
    CosineSimilarity is adopted by default.
        EXAMPLE:
            Sets the dictionary containing only one field (Plot) with one representation, CosineSimilarity as similarity
            measure and the threshold as 1
                CentroidVectorRecommender(item_field={'Plot' : '1'}, similarity=CosineSimilarity(), threshold=1)

    Args:
        item_field (dict): dictionary containing the field names and representations to consider of the items
        similarity (Similarity): kind of similarity to use
        threshold (int): threshold for the ratings. If the rating is greater than the threshold, it will be considered
            as positive
    """

    def __init__(self, item_field: dict, similarity: Similarity = CosineSimilarity(), threshold: int = -1):
        super().__init__(item_field, threshold)
        self.__centroid_vector = CentroidVector(similarity)

    def __calc_positive_rated_baglist(self, rated_items: list, ratings: pd.DataFrame):
        """
        Private functions that extracts features from positive rated items available locally.

        Checks every item available locally and, if the score given by the user to said item is bigger
        than the threshold, it extracts its features, stores them into a list and adds it to another
        list that keeps all the features of the positive rated items.
        If there are no ratings or there are no positively rated items an exception is thrown.

        Args:
            rated_items (list): rated items by the user available locally
            ratings (Dataframe): dataframe which contains ratings given by the user

        Returns:
            positive_features_bag_list (list): list containing all the features extracted from the
                positive rated items
        """
        positive_features_bag_list = []
        for item in rated_items:
            single_item_bag_list = []

            if item is not None:
                for item_field in self.item_field:
                    field_representations = self.item_field[item_field]

                    if isinstance(field_representations, str):
                        # We have only one representation
                        representation = field_representations
                        item_representation = item.get_field(item_field).get_representation(representation)
                        CentroidVectorRecommender.__check_representation(
                            item_representation, representation, item_field)

                        single_item_bag_list.append(item_representation.value)

                    else:

                        for representation in field_representations:
                            item_representation = item.get_field(item_field).get_representation(representation)
                            CentroidVectorRecommender.__check_representation(
                                item_representation, representation, item_field)

                            single_item_bag_list.append(item_representation.value)

                if float(ratings[ratings['to_id'] == item.content_id].score) >= self.threshold:
                    positive_features_bag_list.append(single_item_bag_list)

        if len(ratings) == 0:
            raise FileNotFoundError("No rated items available locally!\n"
                                    "The score frame will be empty for the user")
        if len(positive_features_bag_list) == 0:
            raise ValueError("There are only negative items available locally!\n"
                             "The score frame will be empty for the user")

        return positive_features_bag_list

    @staticmethod
    def __check_representation(representation, representation_name: str, item_field: str):
        """
        Checks that the passed representations is an embedding (in which case the granularity must be document)
        or a tf-idf vector, otherwise throws an exception because in these scenarios the centroid calculation
        cannot be computed

        Args:
            representation: representation instance
            representation_name (str): name of the item representation
            item_field (str): name of the field that has said representation
        """
        if not isinstance(representation, EmbeddingField) and \
                not isinstance(representation, FeaturesBagField):
            raise ValueError(
                "The representation %s for the %s field is not an embedding or a tf-idf vector"
                % (representation_name, item_field))

        if isinstance(representation, EmbeddingField):
            if len(representation.value.shape) != 1:
                raise ValueError(
                    "The representation %s for the %s field is not a document embedding, "
                    "so the centroid cannot be calculated" % (representation_name, item_field))

    def predict(self, user_id: str, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None) -> pd.DataFrame:
        """
        After computing the centroid of the positive rated items by the user and getting the similarity scores
        of said centroid compared with every unrated item (using the CentroidVector object), creates and
        returns a recommendation list of unrated items ordered by their similarity score with the centroid

            EXAMPLE:
                Creates a recommendation list of length 1 with the similarity to the centroid as score, only considering
                the item tt0114319 instead of all the unrated items. (Ratings is a DataFrame containing the ratings
                given by the user with id A000)
                    predict(user_id='A000', ratings=ratings, recs_number=1, items_directory='.../somedir',
                    candidate_item_id_list=['tt0114319'])

        Args:
            candidate_item_id_list (list): list of the items that can be recommended, if None
                all unrated items will be used
            user_id (str): user for which compute the prediction
            recs_number (int): how long the ranking will be
            ratings (pd.DataFrame): ratings of a user
            items_directory (str): name of the directory where the items are stored.

        Returns:
             scores (pd.DataFrame): DataFrame whose columns are the ids of the items (to_id),
                and the similarities between the items and the centroid (rating)
        """

        # Loads the items and extracts features from the unrated items, then
        # extracts features from the positive rated items
        # If exception, returns an empty score_frame
        try:
            rated_items, unrated_items, unrated_features_bag_list = \
                super().preprocessing(items_directory, ratings, candidate_item_id_list)
            positive_rated_features_bag_list = self.__calc_positive_rated_baglist(rated_items, ratings)
        except(ValueError, FileNotFoundError) as e:
            logger.warning(str(e))
            columns = ["to_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)
            return score_frame

        logger.info("Computing rated items centroid")
        self.__centroid_vector.compute_centroid(positive_rated_features_bag_list)

        columns = ["to_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)

        logger.info("Computing similarity between centroid and unrated items")
        similarities = self.__centroid_vector.compute_similarities(unrated_features_bag_list)
        for item, similarity in zip(unrated_items, similarities):
            if item is not None:
                score_frame = pd.concat(
                    [score_frame,
                     pd.DataFrame.from_records([(item.content_id, similarity)], columns=columns)],
                    ignore_index=True)

        score_frame = score_frame.sort_values(['rating'], ascending=False).reset_index(drop=True)
        score_frame = score_frame[:recs_number]

        return score_frame
