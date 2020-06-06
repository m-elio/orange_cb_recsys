import os
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree

from orange_cb_recsys.content_analyzer.content_representation.content import Content
from orange_cb_recsys.recsys.score_prediction_algorithms.score_prediction_algorithm import RatingsSPA

import pandas as pd


class CentroidVector(RatingsSPA):
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def predict(self, item: Content, ratings: pd.DataFrame, items_directory: str, item_to_classify):
        pass


class ClassifierRecommender(RatingsSPA):
    def __init__(self, item_field: str, field_representation: str):
        super().__init__(item_field, field_representation)

    def predict(self, item: Content, ratings: pd.DataFrame, items_directory: str, item_to_classify):
        items = [filename for filename in os.listdir(items_directory)]

        features_bag_list = []
        rated_item_index_list = []
        for item in items:
            item_filename = items_directory + '/' + item
            with open(item_filename, "rb") as content_file:
                content = pickle.load(content_file)

                features_bag_list.append(content.get_field("Plot").get_representation("1").get_value())
        features_bag_list.append(content.get_field("Plot").get_representation("1").get_value())
        v = DictVectorizer(sparse=False)

        X_tmp = v.fit_transform(features_bag_list)

        for i in X_tmp:
            if X_tmp[i].get_content_id() in ratings.item_id:
                rated_item_index_list.append(X_tmp[i])

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(rated_item_index_list, ratings.score)

        return clf.predict(item_to_classify)
