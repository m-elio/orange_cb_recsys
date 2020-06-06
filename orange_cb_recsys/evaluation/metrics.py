import time
from typing import Dict
import pandas as pd
import os

from orange_cb_recsys.evaluation.ranking_metrics import *
from orange_cb_recsys.evaluation.prediction_metrics import *
from orange_cb_recsys.evaluation.fairness_metrics import *


def perform_ranking_metrics(predictions: pd.DataFrame,
                            truth: pd.DataFrame,
                            **options) -> Dict[str, float]:
    """
    Perform the computation of all ranking metrics

    Args:
        predictions (pd.DataFrame): each row contains index(the rank position), label, value predicted
        truth (pd.DataFrame): the real rank each row contains index(the rank position), label, value
        **options : you can specify some option parameters like:
         - fn (int): the n of the Fn metric, default = 1

    Returns:
        results (Dict[str, object]): results of the computations of all ranking metrics
    """
    content_prediction = pd.Series(predictions['to_id'].values)
    if "relevant_threshold" in options.keys():
        relevant_rank = truth[truth['rating'] >= options["relevant_threshold"]]
    else:
        relevant_rank = truth

    content_truth = pd.Series(relevant_rank['to_id'].values)

    results = {
        "Precision": perform_precision(prediction_labels=content_prediction, truth_labels=content_truth),
        "Recall": perform_recall(prediction_labels=content_prediction, truth_labels=content_truth),
        "MRR": perform_MRR(prediction_labels=content_prediction, truth_labels=content_truth),
        "NDCG": perform_NDCG(predictions=predictions, truth=truth),
        "pearson": perform_correlation(prediction_labels=content_prediction, truth_labels=content_truth),
        "kendall": perform_correlation(prediction_labels=content_prediction, truth_labels=content_truth,
                                       method='kendall'),
        "spearman": perform_correlation(prediction_labels=content_prediction, truth_labels=content_truth,
                                        method='spearman'),
    }

    if "fn" in options.keys() and options["fn"] > 1:
        results["F{}".format(options["fn"])] = perform_Fn(n=options["fn"], precision=results["Precision"],
                                                          recall=results["Recall"])
    else:
        results["F1"] = perform_Fn(precision=results["Precision"], recall=results["Recall"])

    return results


def perform_fairness_metrics(score_frame: pd.DataFrame, truth_frame: pd.DataFrame, file_output_directory: str = None,
                             algorithm_name: str = None) -> (pd.DataFrame, pd.DataFrame):
    if file_output_directory is None:
        file_output_directory = 'datasets/evaluation'
    path = "{}/{}/{}_{}".format(os.getcwd(), file_output_directory, algorithm_name, str(int(time.time())))
    try:
        os.mkdir(path)
    except OSError:
        path = "../../{}/{}_{}".format(file_output_directory, algorithm_name, str(int(time.time())))
        try:
            with open(path):
                pass

        except FileNotFoundError:
            path = "{}/{}_{}".format(file_output_directory, algorithm_name, str(int(time.time())))
    print("working in dir: {}".format(path))

    pop_items = popular_items(score_frame=score_frame)
    pop_ratio_user = pop_ratio_by_user(score_frame=score_frame, pop_items=pop_items)

    user_groups = split_user_in_groups(score_frame=score_frame, groups={'niche': 0.2, 'diverse': 0.6,
                                                                        'bb_focused': 0.2}, pop_items=pop_items)
    df_gini = perform_gini_index(score_frame=score_frame)
    delta_gap_score = perform_delta_gap(score_frame=score_frame, truth_frame=truth_frame, users_groups=user_groups)
    profile_vs_recs_pop_ratio = perform_pop_ratio_profile_vs_recs(user_groups=user_groups, truth_frame=truth_frame,
                                                                  most_popular_items=pop_items,
                                                                  pop_ratio_by_users=pop_ratio_user)

    #results_by_user = pd.merge(df_gini, on='from_id')
    results_by_user = df_gini
    results_by_user_group = pd.merge(delta_gap_score, profile_vs_recs_pop_ratio, on='user_group')

    print(results_by_user)
    print(results_by_user_group)
    catalog_coverage(score_frame=score_frame, truth_frame=truth_frame)
    return results_by_user, results_by_user_group


def perform_prediction_metrics(predictions: pd.Series, truth: pd.Series) -> Dict[str, object]:
    """
    Performs the metrics for evaluating the rating prediction phase and returns their values

    Args:
        predictions (pd.Series): Series containing the predicted ratings
        truth (pd.Series): Series containing the truth rating values

    Returns:
        results (Dict[str, object]): Python dictionary where the keys are the names of the metrics and the
        values are the corresponding values
    """
    results = {
        "RMSE": perform_rmse(predictions, truth),
        "MAE": perform_mae(predictions, truth)
    }
    return results
