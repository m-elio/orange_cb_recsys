import networkx as nx
from abc import ABC, abstractmethod
from typing import List

from orange_cb_recsys.recsys.graphs.graph import FullGraph, PropertyNode


class FeatureSelection(ABC):
    """
    Feature selection algorithm class. It has only one abstract method perform which is used to perform
    Feature Selection
    """
    @abstractmethod
    def perform(self, X, y):
        raise NotImplementedError


class FSPageRank(FeatureSelection):
    """
    This class implements PageRank as Feature Selection algorithm. In the case of this project, the nodes of a graph
    can be User, Item or Property (exogenous). In particular, there are several Property nodes and not all of them
    are equally useful. In a FullGraph, each Property node represents a value that refers to a particular Exogenous
    Property, encoded as label in the edge connecting the Property node to the Item or User node (example:
    property node: 'http://dbpedia.org/resource/Phil_Collins', property link label: 'starring'). It would be useful
    to consider only the Property nodes in a graph that refer to the most important labels in order to reduce
    the space complexity of the graph. To do so, Features are represented as Property labels (such as 'starring' or
    'film_director') and the aim is to return a subset of the Property labels in the graph, so that nodes
    that do not refer to this subset of labels may be deleted from the graph

    Args:
        k (int): number of the top k features that will be extracted
    """

    def __init__(self, k):
        super().__init__()
        self.__k = k

    def perform(self, X: FullGraph, y: List[object]) -> List[str]:
        """
        In order to remove features from the graph (which in this case are properties either for user or item nodes),
        a new networkx directed graph is instantiated. This graph will have two types of nodes: the first being either
        item or user nodes, the second being property nodes (but instead of having a value like a DBPedia URI, these
        nodes will have the property label as value). Each item/user node will be connected to the property nodes for
        which they have a valid value in their representation.

        EXAMPLE: the networkx graph will have the following edge:

        tt0112453 (node representing the item tt0112453) -> starring (PropertyNode representing the label 'starring')

        if tt0112453 is connected to a PropertyNode in the original graph that represents a value for the
        label starring (example: in the original graph tt0112453 (ItemNode) -> http://dbpedia.org/resource/Phil_Collins
        (PropertyNode) and the edge between them has the label 'starring').

        After creating this new graph, PageRank will be run and the k properties with the highest value of PageRank
        will be extracted from the graph.

        In case of multiple representations for items or users appropriate measures are adopted in order to merge
        the representations (example: if there are 'starring_0' and 'starring_1' labels, these labels will be merged
        into the 'starring' label and their PageRank value will be summed up)

        Args:
            X (FullGraph): FullGraph representing the graph on which the feature selection technique will be done
            y (List[object]): can be a list containing either items or users in the graph, otherwise an exception is
            thrown. This list represents the target nodes that will be used in the graph to extract their properties.
            This can be useful in case only a subset of items or users is considered (for example, in the project we
            are considering only items not rated by the user)

        Returns:
            new_prop (List[str]): list containing the top k most meaningful property labels
             (example: ['starring', 'producer'])
        """

        if self.__k <= 0:
            return []

        new_graph = nx.DiGraph()

        # checks that all nodes in the target list are either user nodes or item nodes, an exception is thrown otherwise
        if all(X.is_item_node(node) for node in y):
            if X.get_item_exogenous_properties() is not None and self.__k >= len(X.get_item_exogenous_properties()):
                return X.get_item_exogenous_properties()
            representation = X.get_item_exogenous_representation()
        elif all(X.is_user_node(node) for node in y):
            if X.get_user_exogenous_properties() is not None and self.__k >= len(X.get_user_exogenous_properties()):
                return X.get_user_exogenous_properties()
            representation = X.get_user_exogenous_representation()
        else:
            raise ValueError("Target list must contain items or users of the corresponding graph")

        # retrieves the property nodes (successors) from the original graph. For each node in the target list
        # retrieves the data regarding the link between the node in the target list and each property.
        # Adds a new property node to the new graph with value being the link label of the original graph
        for node in y:
            for successor_node in X.get_successors(node):
                if X.is_property_node(successor_node):
                    new_property = X.get_link_data(node, successor_node)
                    new_graph.add_edge(node, PropertyNode(new_property['label']), weight=new_property['weight'])

        # computes PageRank and extracts all properties from it
        rank = nx.pagerank(new_graph.to_undirected())
        rank = {node.value: rank[node] for node in rank if isinstance(node, PropertyNode)}

        # in case multiple representations are considered, the ranking containing multiple representations
        # will be transformed into a ranking containing a single one
        # example: {'starring_0': 0.03, 'starring_1': 0.1, ...}
        # will be transformed into {'starring': 0.13, ...}
        if representation is None:
            new_rank = {}
            properties = set(rank.keys())
            properties = set('_'.join(property_name.split('_')[:-1]) for property_name in properties)
            for property_name in properties:
                properties_labels = [key for key in rank.keys() if property_name in key]
                new_rank[property_name] = 0
                for property_label in properties_labels:
                    new_rank[property_name] += rank[property_label]
            rank = new_rank

        # the ranking produced by the PageRank algorithm is sorted by values and the top k are extracted
        rank = dict(sorted(rank.items(), key=lambda item: item[1], reverse=True))
        rank = list(rank.keys())[:self.__k]

        return rank
