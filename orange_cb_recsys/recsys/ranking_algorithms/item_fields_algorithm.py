from abc import abstractmethod

import pandas as pd
from typing import List
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import hstack

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
from orange_cb_recsys.utils.load_content import get_rated_items, load_content_instance, get_unrated_items


class ItemFieldsRankingAlgorithm(RankingAlgorithm):
    """
    Class that extends RankingAlgorithm. This class represents ranking algorithms that need to know
    each representation to consider of each specified field during the recommendation process.

    Args:
        item_fields(dict): dictionary where the keys are the field names and the
            values are lists containing the representations to consider or strings
            containing a single representation. Example: {'Plot':['0', '1'], 'Genre': '0'}
        threshold(int): if an item rated by the user has a score bigger than
            the threshold, it is considered as a positive example
    """

    def __init__(self, item_fields: dict, threshold: int = 0):
        super().__init__()
        self.__item_fields = item_fields
        self.__threshold = threshold

    @property
    def threshold(self):
        return self.__threshold

    @property
    def item_fields(self):
        return self.__item_fields

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    @item_fields.setter
    def item_fields(self, item_fields):
        self.__item_fields = item_fields

    def __calc_unrated_baglist(self, unrated_items: list):
        """
        Private functions that extracts features from unrated_items available locally.

        If multiple representations of the item are specified in the constructor of the class,
        we extract features from all of them, otherwise if a single representation is specified,
        we extract features from that single representation.

        This method can throw three exceptions. The first is thrown if a field representation is an empty list
        (input example: {'Plot': []}), the second if a representation doesn't exist in the field
        (input example: {'Plot': '10'} but the field plot of the items doesn't have the representation '10'), the third
        if a field specified in item_fields doesn't exist in the items (input example: {'Test': '1'} but 'Test' is not
        a field for any item).

        Args:
            unrated_items (list): unrated items available locally
        Returns:
            unrated_features_bag_list (list): list that contains features extracted
                                        from the unrated_items
        """

        # Since all the items share the same structure, takes the first item in the unrated item list
        # and checks that every field specified in the item_fields is also in the item
        # otherwise an exception is thrown
        first_item = unrated_items[0]
        for field in self.__item_fields.keys():
            if field not in first_item.field_dict.keys():
                raise ValueError("The %s field could not be found in the items" % field)

        unrated_features_bag_list = []

        for item in unrated_items:
            single_item_bag_list = []
            for item_field in self.__item_fields:
                field_representations = self.__item_fields[item_field]
                try:

                    if isinstance(field_representations, str):
                        # We have only one representation
                        representation = field_representations
                        single_item_bag_list.append(
                            item.get_field(item_field).get_representation(representation).value
                        )
                    else:
                        if len(field_representations) == 0:
                            raise ValueError("Cannot compute an empty representation")
                        for representation in field_representations:
                            single_item_bag_list.append(
                                item.get_field(item_field).get_representation(representation).value
                            )

                except KeyError:
                    raise ValueError("The representation %s wasn't found for the %s field" %
                                     (representation, item_field))

            unrated_features_bag_list.append(single_item_bag_list)

        return unrated_features_bag_list

    def preprocessing(self, items_directory: str, ratings: pd.DataFrame, candidate_item_id_list: list = None):
        """
        Function used to retrieve data that will be used in the computation of the ranking.
        It loads the rated and unrated items, computes the threshold if it was set to -1 and
        extracts the features from the unrated items.

        This method can throw two exceptions. The first one is thrown if the threshold value specified
        in the constructor of the class it's not in the range [-1, 1], the second one is thrown if,
        while considering a candidate_item_id_list passed as an argument, there are no valid
        items to consider (example: ['test', 'test2'] but neither test nor test2 are items in the
        items directory)

        Args:
            items_directory (str): directory where the items are stored
            ratings (Dataframe): dataframe which contains ratings given by the user
            candidate_item_id_list (list): list of the items that can be recommended, if None
            all unrated items will be used

        Returns:
            rated_items (list): list containing the instances of the rated items
            unrated_items (list): list containing the instances of the unrated items
            unrated_features_baglist (list): list containing the features extracted from the unrated items
        """

        # If threshold is the min possible (range is [-1, 1]), we calculate the mean value
        # of all the ratings and set it as the threshold. Also an exception is thrown if the
        # threshold value is not in the range
        if not -1 <= self.__threshold <= 1:
            raise ValueError("Threshold value must be in the range [-1, 1]")

        if self.__threshold == -1:
            self.__threshold = pd.to_numeric(ratings["score"], downcast="float").mean()

        # Load unrated items from the path
        if candidate_item_id_list is None or len(candidate_item_id_list) == 0:
            unrated_items = get_unrated_items(items_directory, ratings)
        else:
            # If a candidate list is specified, it loads only items that are valid (it doesn't add None to the list)
            unrated_items = [load_content_instance(items_directory, item_id) for item_id in candidate_item_id_list
                             if load_content_instance(items_directory, item_id) is not None]

        if len(unrated_items) == 0:
            raise ValueError("No valid unrated items found")

        # Load rated items from the path
        rated_items = get_rated_items(items_directory, ratings)

        return rated_items, unrated_items, self.__calc_unrated_baglist(unrated_items)

    @abstractmethod
    def predict(self, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        raise NotImplementedError


def transform(transformer: DictVectorizer, X: list):
    """
    Transform the X passed vectorizing if X contains dicts and merging
    multiple representations in a single one for every item in X.
    So if X = [
                [dict, arr, arr]
                    ...
                [dict, arr, arr]
            ]
    where every sublist contains multiple representation for a single item,
    the function returns:
    X = [
            arr,
            ...
            arr
        ]
    Where every row is the fused representation for the item

    Args:
        X (list): list that contains representations of the items
        transformer (DictVectorizer): DictVectorizer object from the sklearn library

    Returns:
        X fused and vectorized

    """

    # We check if there are dicts as representation in the first element of X,
    # since the representations are the same for all elements in X we can check
    # for dicts only in one element
    need_vectorizer = any(isinstance(rep, dict) for rep in X[0])

    if need_vectorizer:
        # IF the transformer is not fitted then we are training the model
        try:
            check_is_fitted(transformer)
        except NotFittedError:
            X_dicts = []
            for item in X:
                for rep in item:
                    if isinstance(rep, dict):
                        X_dicts.append(rep)

            transformer.fit(X_dicts)

        # In every case, we transform the input
        X_vectorized = []
        for sublist in X:
            single_list = []
            for item in sublist:
                if isinstance(item, dict):
                    vector = transformer.transform(item)
                    single_list.append(vector)
                else:
                    single_list.append(item)
            X_vectorized.append(single_list)
    else:
        X_vectorized = X
    try:
        X_sparse = [hstack(sublist).toarray().flatten() for sublist in X_vectorized]
    except ValueError:
        X_sparse = [np.column_stack(sublist).flatten() for sublist in X_vectorized]

    return X_sparse
