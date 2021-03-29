from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from orange_cb_recsys.content_analyzer.content_representation.content import Content


class Algorithm(ABC):
    """
    Abstract class for the algorithms
    """
    def __init__(self):
        super().__init__()


class RankingAlgorithm(Algorithm):
    """
    Abstract class for the ranking algorithms
    """
    @abstractmethod
    def predict(self, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        """
        Args:
            candidate_item_id_list: list of the items that can be recommended, if None
                all unrated items will be used
            recs_number (int): How long the ranking will be
            ratings (pd.DataFrame): ratings of a specific user
            items_directory (str): Name of the directory where the items are stored.
        """
        raise NotImplementedError


class ScorePredictionAlgorithm(Algorithm):
    """
    Abstract class for the score prediction algorithms
    """
    @abstractmethod
    def predict(self, user_id: str, items: List[Content], ratings: pd.DataFrame, items_directory: str):
        raise NotImplementedError
