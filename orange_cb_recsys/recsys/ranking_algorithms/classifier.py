import collections
from abc import ABC
from typing import List

from sklearn import neighbors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

from orange_cb_recsys.recsys.ranking_algorithms.item_fields_algorithm import\
    ItemFieldsRankingAlgorithm, transform
from orange_cb_recsys.utils.const import logger


class Classifier(ABC):
    """
    Abstract class for Classifiers.
    It keeps all the attributes that the classifiers need: their parameters, the instantiated classifier
    and the pipeline.

    It has an abstract fit() method and an abstract predict_proba() method.
    """

    def __init__(self, **classifier_parameters):
        super().__init__()
        self.__classifier_parameters = classifier_parameters
        self.__empty_parameters = False
        if len(classifier_parameters) == 0:
            self.__empty_parameters = True
        self.__clf = None
        self.__pipe = None
        self.__transformer = DictVectorizer(sparse=True, sort=False)

    @property
    def classifier_parameters(self):
        return self.__classifier_parameters

    @property
    def empty_parameters(self):
        return self.__empty_parameters

    @property
    def clf(self):
        return self.__clf

    @clf.setter
    def clf(self, clf):
        self.__clf = clf

    @property
    def pipe(self):
        return self.__pipe

    @pipe.setter
    def pipe(self, pipe):
        self.__pipe = pipe

    @property
    def transformer(self):
        return self.__transformer

    def fit(self, X: list, Y: list = None):
        """
        Fit the classifier.
        First the classifier is instantiated, then we transform the Training Data,
        then the actual fitting is done.

        Training data (X) is in the form:
            X = [ [representation1, representation2], [representation1, representation2], ...]
        where every sublist contains the representation chosen of the chosen fields for a item.

        Target data (Y) is in the form:
            Y = [0, 1, ... ]

        Args:
            X (list): list containing Training data.
            Y (list): list containing Training targets.
        """
        raise NotImplementedError

    def predict_proba(self, X_pred: list):
        """
        Predicts the probability for every item in X_pred.
        First we transform the data, then the actual prediction is done.
        It uses the method predict_proba() from sklearn of the instantiated classifier

        It's in the form:
            X_pred = [ [representation1, representation2], [representation1, representation2], ...]
        where every sublist contains the representation chosen of the chosen fields for a item.

        Args:
            X_pred (list): list containing data to predict.
        """
        raise NotImplementedError


class KNN(Classifier):
    """
    Class that implements the KNN Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the Classifier directly from sklearn
    """
    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)

    def __instantiate_classifier(self, X: list):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.

        Since KNN has n_neighbors = 5 as default, it can throw an exception if less sample in
        the training data are provided, so we change dynamically the n_neighbors parameter
        according to the number of samples

        Args:
            X (list): Training data
        """
        if self.empty_parameters:
            if len(X) < 5:
                self.clf = neighbors.KNeighborsClassifier(n_neighbors=len(X))
            else:
                self.clf = neighbors.KNeighborsClassifier()
        else:
            self.clf = neighbors.KNeighborsClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier(X)

        # Transform the input if there are dicts, multiple representation, etc.
        X = transform(self.transformer, X)

        pipe = make_pipeline(self.clf)
        self.pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):

        X_pred = transform(self.transformer, X_pred)

        return self.pipe.predict_proba(X_pred)


class RandomForest(Classifier):
    """
    Class that implements the Random Forest Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.empty_parameters:
            self.clf = RandomForestClassifier(n_estimators=400, random_state=42)
        else:
            self.clf = RandomForestClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = transform(self.transformer, X)

        pipe = make_pipeline(self.clf)
        self.pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = transform(self.transformer, X_pred)

        return self.pipe.predict_proba(X_pred)


class SVM(Classifier):
    """
    Class that implements the SVM Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier SVC directly from sklearn.

    The particularity is that if folds can be executed and are calculated with the
    method calc_folds(), then a Calibrated SVC classifier is instantiated.
    Otherwise a simple SVC classifier is instantiated.
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)

    def calc_folds(self, labels: list):
        """
        Private functions that check what number of folds should SVM classifier do.

        By default SVM does 5 folds, so if there are less ratings we decrease the number of
        folds because it would throw an exception otherwise.
        Every class should have min 2 rated items, otherwise no folds can be executed.

        EXAMPLE:
                labels = [1 1 0 1 0]

            We count how many different values there are in the list with
            collections.Counter(labels), so:
                count = {"1": 3, "0": 2} # There are 3 rated_items of class 1
                                        # and 2 rated_items of class 0

            Then we search the min value in the dict with min(count.values()):
                min_fold = 2

        Args:
            labels: list of labels of the rated_items
        Returns:
            Number of folds to do.

        """
        count = collections.Counter(labels)
        min_fold = min(count.values())

        if min_fold < 2:
            logger.warning("There's too few rating for a class! There needs to be at least 2!\n"
                           "No folds will be executed")
        elif min_fold >= 5:
            min_fold = 5

        self.__folds = min_fold

    def __instantiate_classifier(self, calibrated: bool = True):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.

        Args:
            calibrated (bool): If True, a calibrated svc classifier is instantiated.
                Otherwise, a non-calibrated svc classifier is instantiated
        """

        if calibrated:
            if self.empty_parameters:
                self.clf = CalibratedClassifierCV(
                    SVC(kernel='linear', probability=True),
                    cv=self.__folds)

            else:
                self.clf = CalibratedClassifierCV(
                    SVC(kernel='linear', probability=True, **self.classifier_parameters),
                    cv=self.__folds)
        else:
            if self.empty_parameters:
                self.clf = SVC(kernel='linear', probability=True)
            else:
                self.clf = SVC(kernel='linear', probability=True, **self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        # Transform the input
        X = transform(self.transformer, X)

        # Try fitting the Calibrated classifier for better classification
        try:
            self.__instantiate_classifier(calibrated=True)

            pipe = make_pipeline(self.clf)
            self.pipe = pipe.fit(X, Y)

        # If exception instantiate a non-calibrated classifier, then fit
        except ValueError:
            self.__instantiate_classifier(calibrated=False)

            pipe = make_pipeline(self.clf)
            self.pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = transform(self.transformer, X_pred)

        return self.pipe.predict_proba(X_pred)


class LogReg(Classifier):
    """
    Class that implements the Logistic Regression Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.empty_parameters:
            self.clf = LogisticRegression(random_state=42)
        else:
            self.clf = LogisticRegression(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = transform(self.transformer, X)

        pipe = make_pipeline(self.clf)
        self.pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = transform(self.transformer, X_pred)

        return self.pipe.predict_proba(X_pred)


class DecisionTree(Classifier):
    """
    Class that implements the Decision Tree Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.empty_parameters:
            self.clf = DecisionTreeClassifier(random_state=42)
        else:
            self.clf = DecisionTreeClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = transform(self.transformer, X)

        pipe = make_pipeline(self.clf)
        self.pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = transform(self.transformer, X_pred)

        return self.pipe.predict_proba(X_pred)


class GaussianProcess(Classifier):
    """
    Class that implements the Gaussian Process Classifier from sklearn.
    The parameters one could pass are the same ones you would pass instantiating
    the classifier directly from sklearn
    """

    def __init__(self, **classifier_parameters):
        super().__init__(**classifier_parameters)

    def __instantiate_classifier(self):
        """
        Instantiate the sklearn classifier.

        If parameters were passed in the constructor of this class, then
        we pass them to sklearn.
        """
        if self.empty_parameters:
            self.clf = GaussianProcessClassifier(random_state=42)
        else:
            self.clf = GaussianProcessClassifier(**self.classifier_parameters)

    def fit(self, X: list, Y: list = None):

        self.__instantiate_classifier()

        X = transform(self.transformer, X)

        pipe = make_pipeline(self.clf)
        self.pipe = pipe.fit(X, Y)

    def predict_proba(self, X_pred: list):
        X_pred = transform(self.transformer, X_pred)

        return self.pipe.predict_proba(X_pred)


class ClassifierRecommender(ItemFieldsRankingAlgorithm):
    """
       Class that implements recommendation through a specified Classifier.
       In the constructor must be specified parameter needed for the recommendations.
       To effectively get the recommended items, the method predict() of this class
       must be called after instantiating the ClassifierRecommender.
        EXAMPLE:
            # Interested in only a field representation, DecisionTree classifier,
            # threshold = 0
             alg = ClassifierRecommender({"Plot": "0"}, DecisionTree(), 0)

            # Interested in only a field representation, KNN classifier with custom parameter,
            # threshold = 0
             alg = ClassifierRecommender({"Plot": "0"}, KNN(n_neighbors=3), 0)

            # Interested in multiple field representations of the items, KNN classifier with custom parameter,
            # threshold = 0
             alg = ClassifierRecommender(
                                    item_field={"Plot": ["0", "1"],
                                                "Genre": ["0", "1"],
                                                "Director": ["1"]},
                                    classifier=KNN(n_neighbors=3),
                                    threshold=0 )

            # After instantiating the ClassifierRecommender, call the method predict to get
            # recommendations. Check the predict() method documentation for more
             alg.predict('U1', rating, 1, path)

       Args:
           item_fields (dict): dict where the key is the name of the field
                that contains the content to use, value is the representation(s) that will be
                used for the said item. The value of a field can be a string or a list,
                use a list if you want to use multiple representations for a particular field.
                EXAMPLE:
                    {'Plot': '0'}
                    {'Genre': '1'}
                    {'Plot': ['0','1'], 'Genre': '0', 'Director': ['0', '1', '2']}
           classifier (Classifier): classifier that will be used
               can be one object of the Classifier class.
           threshold (int): ratings bigger than threshold will be
               considered as positive
       """

    def __init__(self, item_fields: dict, classifier: Classifier, threshold: int = -1):
        super().__init__(item_fields, threshold)
        self.__classifier = classifier

    def __calc_labels_rated_baglist(self, rated_items: list, ratings: pd.DataFrame):
        """
        Private functions that calculates labels of rated_items available locally and
        extracts features from them.

        For every rated_items available locally, if the rating given is >= threshold
        then we label it as 1, 0 otherwise.
        We also extract features from the rated items that we will use later to fit the
        classifier, from a single representation or from multiple ones.
        IF there are no rated_items available locally or if there are only positive/negative
        items, an exception is thrown.

        Args:
            rated_items (list): rated items by the user available locally
            ratings (DataFrame): Dataframe which contains ratings given by the user
        Returns:
            labels (list): list of labels of the rated items
            rated_features_bag_list (list): list that contains features extracted
                                    from the rated_items
        """
        labels = []
        rated_features_bag_list = []

        for item in rated_items:
            if item is not None:
                single_item_bag_list = []
                for item_field in self.item_fields:
                    field_representations = self.item_fields[item_field]
                    if isinstance(field_representations, str):
                        # We have only one representation
                        representation = field_representations
                        single_item_bag_list.append(
                            item.get_field(item_field).get_representation(representation).value
                        )
                    else:
                        for representation in field_representations:
                            single_item_bag_list.append(
                                item.get_field(item_field).get_representation(representation).value
                            )
                labels.append(
                        1 if float(ratings[ratings['to_id'] == item.content_id].score) >= self.threshold else 0
                )
                rated_features_bag_list.append(single_item_bag_list)

        if len(labels) == 0:
            raise FileNotFoundError("No rated item available locally!\n"
                                    "The score frame will be empty for the user")
        if 0 not in labels:
            raise ValueError("There are only positive items available locally!\n"
                             "The score frame will be empty for the user")
        elif 1 not in labels:
            raise ValueError("There are only negative items available locally!\n"
                             "The score frame will be empty for the user")

        return labels, rated_features_bag_list

    def predict(self, ratings: pd.DataFrame, recs_number: int, items_directory: str,
                candidate_item_id_list: List = None) -> pd.DataFrame:
        """
        Get recommendations for a specified user.

        You must pass the the DataFrame which contains the ratings of the user, how many
        recommended item the method predict() must return, and the path of the items.
        If recommendation for certain item is needed, specify them in candidate_item_id_list
        parameter. In this case, the recommender system will return only scores for the items
        in the list, ignoring the recs_number parameter.
         EXAMPLE
            # Instantiate the ClassifierRecommender object, check its documentation if needed
             alg = ClassifierRecommender(...)

            # Get 5 most recommended items for the user 'AOOO'
             alg.predict('A000', rat, 5, path)

            # Get the score for the item 'tt0114576' for the user 'A000'
             alg.predict('A000', ratings, 1, path, ['tt0114576'])

        Args:
            ratings (pd.DataFrame): ratings of the user with id equal to user_id
            recs_number (int): How long the ranking will be
            items_directory (str): Path to the directory where the items are stored.
            candidate_item_id_list: list of the items that can be recommended, if None
            all unrated items will be used
        Returns:
            The predicted classes, or the predict values.
        """

        # Loads the items and extracts features from the unrated items, then
        # calculates labels and extracts features from the rated items
        # If exception, returns an empty score_frame
        try:
            rated_items, unrated_items, unrated_features_bag_list = \
                super().preprocessing(items_directory, ratings, candidate_item_id_list)
            labels, rated_features_bag_list = self.__calc_labels_rated_baglist(rated_items, ratings)
        except(ValueError, FileNotFoundError) as e:
            logger.warning(str(e))
            columns = ["to_id", "rating"]
            score_frame = pd.DataFrame(columns=columns)
            return score_frame

        # If the classifier chosen is SVM we calc how many folds the classifier
        # can do. If no folds is possible, no folds will be executed
        if isinstance(self.__classifier, SVM):
            self.__classifier.calc_folds(labels)

        self.__classifier.fit(rated_features_bag_list, labels)

        columns = ["to_id", "rating"]
        score_frame = pd.DataFrame(columns=columns)

        logger.info("Predicting scores")
        score_labels = self.__classifier.predict_proba(unrated_features_bag_list)

        for score, item in zip(score_labels, unrated_items):
            score_frame = pd.concat(
                [score_frame, pd.DataFrame.from_records([(item.content_id, score[1])], columns=columns)],
                ignore_index=True)

        score_frame = score_frame.sort_values(['rating'], ascending=False).reset_index(drop=True)
        score_frame = score_frame[:recs_number]

        return score_frame
