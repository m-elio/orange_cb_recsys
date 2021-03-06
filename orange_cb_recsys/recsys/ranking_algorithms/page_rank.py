from typing import List, Dict
import networkx as nx
import pandas as pd
from abc import abstractmethod
from copy import deepcopy

from orange_cb_recsys.recsys.algorithm import RankingAlgorithm
from orange_cb_recsys.recsys.graphs.full_graphs import NXFullGraph
from orange_cb_recsys.recsys.graphs.graph import FullGraph
from orange_cb_recsys.utils.const import logger
from orange_cb_recsys.utils.feature_selection import FeatureSelection


class PageRankAlg(RankingAlgorithm):
    """
    Abstract class that contains the main methods and attributes for any PageRank algorithm. Considering a FullGraph
    (which is a graph with User, Item and Property nodes) the class aims to provide a method to perform
    PageRank on this graph and returning a recommendation list based on the produced ranking. The final
    recommendation list can be personalized by removing certain nodes. The most important method is perform
    which encapsulates the PageRank computation. It is an abstract method so that the calculation can be
    personalized

    Args:
        fullgraph (FullGraph): original graph on which the PageRank or Feature Selection algorithms will be applied.
            Note that it's useful to define copies of this graph on which apply modifications and not to operate on the
            original instance itself. This is so because if any iterative external operation has to be done (like
            using the PageRankAlg's predict method two times in a row with different parameters) the original fullgraph
            has to be preserved, otherwise each successive operation will be influenced by the previous ones
        remove_user_nodes (bool): If True, removes user nodes from the ranking
        remove_items_in_profile (bool): If True, removes item nodes from the ranking that are also
            in the user profile
        remove_properties (bool): If True, removes property nodes from the ranking
    """
    def __init__(self, fullgraph: FullGraph = None, remove_user_nodes: bool = True,
                 remove_items_in_profile: bool = True, remove_properties: bool = True):
        super().__init__()
        self.__fullgraph = fullgraph
        self.__remove_user_nodes: bool = remove_user_nodes
        self.__remove_items_in_profile: bool = remove_items_in_profile
        self.__remove_properties: bool = remove_properties

    @property
    def fullgraph(self) -> FullGraph:
        """
        Getter for the graph on which the PageRank algorithm will be executed
        """
        return self.__fullgraph

    @fullgraph.setter
    def fullgraph(self, graph: FullGraph):
        """
        Sets the FullGraph on which the PageRank algorithm will be executed
        """
        self.__fullgraph = graph

    @property
    def remove_user_nodes(self) -> bool:
        """
        Getter for the boolean attribute remove_user_nodes
        """
        return self.__remove_user_nodes

    @remove_user_nodes.setter
    def remove_user_nodes(self, remove_user_nodes: bool):
        """
        Setter for the boolean attribute remove_user_nodes
        True if you want to remove nodes that refer to users when extracting a user profile, False otherwise
        """
        self.__remove_user_nodes = remove_user_nodes

    @property
    def remove_items_in_profile(self) -> bool:
        """
        Getter for the boolean attribute remove_items_in_profile
        """
        return self.__remove_items_in_profile

    @remove_items_in_profile.setter
    def remove_items_in_profile(self, remove_items_in_profile: bool):
        """
        Setter for the boolean attribute remove_items_in_profile
        True if you want to remove item nodes that are in the user ratings, False otherwise
        """
        self.__remove_items_in_profile = remove_items_in_profile

    @property
    def remove_properties(self) -> bool:
        """
        Getter for the boolean attribute remove_properties
        """
        return self.__remove_properties

    @remove_properties.setter
    def remove_properties(self, remove_properties: bool):
        """
        Setter for the boolean attribute remove_properties
        True if you want to remove property nodes, False otherwise
        """
        self.__remove_properties = remove_properties

    @abstractmethod
    def predict(self, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None):
        """
        Abstract method that should encapsulate the computation process for the PageRank
        """
        raise NotImplementedError

    def clean_rank(self, rank: Dict, graph: FullGraph, user_id: str = None) -> Dict:
        """
        Cleans a rank from all the nodes that are not requested. It's possible to remove user nodes,
        property nodes and item nodes, the latter if they are already in the user profile. This produces a filtered
        ranking with only the desired nodes inside of it. What is filtered depends by the
        attributes remove_user_nodes, remove_items_in_profile and remove_properties

        Args:
            rank (dict): dictionary representing the ranking (keys are nodes and values are their ranked score)
            graph (FullGraph): graph from which the user profile will be extracted
            user_id (str): id of the user used to extract his profile (if None the profile will be empty)

        Returns:
            new_rank (dict): dictionary representing the filtered ranking
        """
        if user_id is not None:
            extracted_profile = self.extract_profile(user_id, graph)
        else:
            extracted_profile = {}

        new_rank = {k: rank[k] for k in rank.keys()}
        for k in rank.keys():
            if self.__remove_user_nodes and graph.is_user_node(k):
                new_rank.pop(k)
            elif self.__remove_items_in_profile and graph.is_item_node(k) and k in extracted_profile.keys():
                new_rank.pop(k)
            elif self.__remove_properties and graph.is_property_node(k):
                new_rank.pop(k)

        return new_rank

    @staticmethod
    def extract_profile(user_id: str, graph: FullGraph) -> Dict:
        """
        Extracts the user profile by accessing the node inside of the graph representing the user.
        Retrieves the item nodes to which the user gave a rating and returns a dictionary containing
        the successor nodes as keys and the weights in the graph for the edges between the user node
        and his successors as values

        Args:
            user_id (str): id for the user for which the profile will be extracted
            graph (FullGraph): graph from which the user profile will be extracted. In particular, the weights
                of the links connecting the user node representing the item and the successors will be
                extracted and will represent the values in the profile dictionary. A graph is passed instead
                of using the original graph in the class because the original graph isn't modified, so it isn't
                affected by modifications done during the prediction process (such as Feature Selection)

        Output example: if the user has rated two items ('I1', 'I2'), the user node corresponding to the user_id
        is selected (for example for user 'A') and each link connecting the user to the items is retrieved and the
        weight of said edge is extracted and added to the dictionary. If the weights of the edges A -> I1 and
        A -> I2 are respectively 0.2 and 0.4 the output will be a dictionary in the following form:
        {'I1': 0.2, 'I2': 0.4}

        Returns:
            profile (dict): dictionary with item successor nodes to the user as keys and weights of the edge
                connecting them in the graph as values
        """
        successors = graph.get_successors(user_id)
        profile = {}
        for successor in successors:
            link_data = graph.get_link_data(user_id, successor)
            profile[successor] = link_data['weight']
            logger.info('unpack %s, %s', str(successor), str(profile[successor]))
        return profile  # {t: w for (f, t, w) in adj}

    @staticmethod
    def remove_links_for_user(graph: FullGraph, nodes_to_remove: set, user_id: str):
        """
        Removes the links between the user node which represents the user_id passed as an argument and a subset of its
        successors defined in the nodes_to_remove argument. After this phase, any node in the graph without any
        predecessor is removed (meaning both the items and the property nodes that may not have any predecessor after
        removing one or more item nodes). This is useful in case a prediction considering only a subset of the items
        an user rated has to be done (in particular this may be the case with Partitioning techniques). If in such cases
        this phase wasn't done and the additional item nodes were simply masked, the results obtained by the prediction
        would be biased by the fact that these nodes and links still exist within the graph

        Args:
            graph (FullGraph): graph on which the links and/or nodes will be removed
            nodes_to_remove (set): set of successor nodes for a specific user for which the links between the user and
                each node will be removed
            user_id (str): string value representing the user_id to consider (used to retrieve the corresponding user
                node from the graph)
        """

        to_remove = set()
        for item_node in graph.get_successors(user_id):
            if item_node in nodes_to_remove:
                to_remove.add((user_id, item_node))
        graph._graph.remove_edges_from(to_remove)

        to_remove = set()
        for item_node in nodes_to_remove:
            if len(graph.get_predecessors(item_node)) == 0:
                to_remove.add(item_node)
                for property_node in graph.get_successors(item_node):
                    if len(graph.get_predecessors(property_node)) == 1:
                        to_remove.add(property_node)
        graph._graph.remove_nodes_from(to_remove)


class NXPageRank(PageRankAlg):
    """
    Algorithm based on the networkx implementation for the FullGraph class. The algorithm also allows to specify
    two types of feature selection algorithms, one for the item nodes and one for the user nodes. The goal is to
    simplify the graph by removing Property nodes that refer to properties ('starring', 'film_director', ...)
    that aren't as meaningful to the ranking as other Property nodes that refer to other properties

    Args:
        graph (NXFullGraph): graph on which the PageRank will be computed
        item_feature_selection_algorithm (FeatureSelection): feature selection algorithm chosen to filter item
            properties
        user_feature_selection_algorithm (FeatureSelection): feature selection algorithm chosen to filter user
            properties
    """
    def __init__(self, graph: NXFullGraph = None, item_feature_selection_algorithm: FeatureSelection = None,
                 user_feature_selection_algorithm: FeatureSelection = None,
                 remove_user_nodes: bool = True, remove_items_in_profile: bool = True, remove_properties: bool = True):
        super().__init__(graph, remove_user_nodes, remove_items_in_profile, remove_properties)
        self.__item_feature_selection_algorithm = item_feature_selection_algorithm
        self.__user_feature_selection_algorithm = user_feature_selection_algorithm

    def predict(self, ratings: pd.DataFrame = None, recs_number: int = 10, items_directory: str = None,
                candidate_item_id_list: List = None):
        """
        Creates a recommendation list containing the top items retrieved by the PageRank algorithm. Networkx provides
        a method to compute PageRank on networkx graphs. Two types of PageRank computations are possible.
        The first one, in case the ranking is made for a user, will be PageRank with Priors considering the user profile
        as personalization vector. The second one, in case no user is defined (empty ratings or None) will be standard
        PageRank.
        If only a subset of the user ratings is passed as an argument, the graph will be pruned from the links
        representing the ratings not considered in said subset.
        For any case in which the graph will be modified (such as Feature Selection), a copy of the original graph will
        be created, so that the original graph may be preserved for future operations.
        It's also possible to include a candidate_item_id_list, in order to consider in the ranking only nodes specified
        in that list.
        Exceptions are thrown if raised by the feature selection algorithms or if a recommendations number <= 0
        is chosen, in these cases an empty recommendation list will be returned.

        Args:
            ratings (pd.Dataframe): ratings of the user for which compute the prediction, if None or empty dataframe
                standard PageRank will be computed instead of personalized PageRank
            recs_number (int): length of the recommendation list
            items_directory (str): not used
            candidate_item_id_list (list): if a candidate list is specified, only items in the candidate list will
                be considered for recommendations (also ignoring the recommendations number)

        Returns:
            score_frame (pd.Dataframe): dataframe containing the recommendation list
        """
        try:
            graph = self.fullgraph

            if recs_number <= 0:
                raise ValueError("You must set a valid number of recommendations (> 0) in order to compute PageRank")

            if candidate_item_id_list is None:
                candidate_item_id_list = []
            if ratings is None:
                ratings = pd.DataFrame()

            if len(ratings) != 0:
                user_id = ratings['from_id'].iloc[0]
                personalized = True

                # in case only a subset of ratings from the user is passed, first of all it checks that
                # the ratings in the dataframe are a subset of the ratings in the graph's user profile
                # this is done to check that there aren't items rated by the user in the dataframe
                # but not in the graph's user profile
                user_ratings = set(ratings['to_id'].values)
                user_graph = set([node for node in graph.get_successors(user_id) if graph.is_item_node(node)])

                if not user_ratings.issubset(user_graph):
                    raise ValueError("There are ratings in the dataframe not available in the graph for the user")

                # after that it check if the ratings in the dataframe are equal to the ratings in the
                # graph's user profile. If they are equal no further operation is done, otherwise
                # the graph is simplified so that only items considered in the dataframe are
                # represented in the graph
                if not user_ratings == user_graph:

                    additional_nodes = user_graph.difference(user_ratings)
                    graph = deepcopy(self.fullgraph)

                    logger.warning("The ratings passed are less than the ratings in the graph's user profile.\n"
                                   "The graph will be pruned in order to consider only the ratings passed")

                    self.remove_links_for_user(graph, additional_nodes, user_id)

            else:
                personalized = False
                user_id = None

            # if the item or the user feature selection algorithms are instantiated it initializes the list of nodes
            # to consider in the feature selection process (which are nodes not referred in the user_ratings, so it
            # doesn't consider the user who the ratings refer to and the items that he voted) and performs the
            # feature selection which will return a list for the new properties to consider (one list for items and
            # one for users)
            if self.__item_feature_selection_algorithm is not None:
                logger.info('Computing feature selection on items')
                if len(ratings) != 0:
                    recommended_items = list(set(ratings['to_id']))
                else:
                    recommended_items = []
                recommended_items = [item for item in graph.item_nodes if item not in recommended_items]
                new_item_prop = self.__item_feature_selection_algorithm.perform(graph, recommended_items)
            else:
                new_item_prop = graph.get_item_exogenous_properties()

            if self.__user_feature_selection_algorithm is not None:
                logger.info('Computing feature selection on users')
                if len(ratings) != 0:
                    recommended_users = list(set(ratings['from_id']))
                else:
                    recommended_users = []
                recommended_users = [user for user in graph.user_nodes if user not in recommended_users]
                new_user_prop = self.__user_feature_selection_algorithm.perform(graph, recommended_users)
            else:
                new_user_prop = graph.get_user_exogenous_properties()

            # the lists created by the feature selection algorithms will be used to remove nodes from the graph so that
            # only the specified user and/or item exogenous properties will be considered
            if self.__user_feature_selection_algorithm is not None or\
                    self.__item_feature_selection_algorithm is not None:

                if graph is self.fullgraph:
                    graph = deepcopy(self.fullgraph)

                nodes_to_remove = set()
                for property_node in graph.property_nodes:
                    for predecessor in graph.get_predecessors(property_node):
                        label = graph.get_link_data(predecessor, property_node)['label']
                        label = '_'.join(label.split('_')[:-1])
                        if (new_item_prop is not None and label not in new_item_prop) and\
                                (new_user_prop is not None and label not in new_user_prop):
                            nodes_to_remove.add(property_node)
                graph._graph.remove_nodes_from(nodes_to_remove)

            # runs the PageRank either the personalized through the user profile or the standard one
            if personalized:
                profile = self.extract_profile(user_id, graph)
                if sum(profile.values()) == 0.0:
                    logger.warning("Cannot compute personalized PageRank if all the weights are the minimum "
                                   "possible value, standard PageRank will be calculated instead")
                    scores = nx.pagerank(graph._graph)
                else:
                    scores = nx.pagerank(graph._graph.to_undirected(), personalization=profile)
            else:
                scores = nx.pagerank(graph._graph)

            # cleans the results removing nodes (they can be user nodes, items in the user profile and properties)
            scores = self.clean_rank(scores, graph, user_id)
            scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
            if len(candidate_item_id_list) == 0:
                ks = list(scores.keys())
                ks = ks[:recs_number]
            else:
                ks = candidate_item_id_list
            new_scores = {k: scores[k] for k in scores.keys() if k in ks}

            columns = ["to_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)

            for item, score in new_scores.items():
                score_frame = pd.concat(
                    [score_frame,
                     pd.DataFrame.from_records([(item.value, score)], columns=columns)],
                    ignore_index=True)

            return score_frame

        except ValueError as e:
            logger.warning(str(e))
            columns = ["to_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)
            return score_frame
