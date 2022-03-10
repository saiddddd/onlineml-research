from tools.data_loader import TimeSeriesDataLoader

from tools.model_trainer_reader import SklearnRandomForestClassifierTrainer, RiverAdaRandomForestClassifier
from tools.model_evaluator import SklearnModelEvaluator, RiverModelEvaluator

from tools.model_perform_visualization import TrendPlot, PredictionProbabilityDist, RocCurve

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc, roc_auc_score

import datetime

import json
import numpy as np
import pandas as pd


#=============================================================#
# Definition of functions for following data processing steps #
#=============================================================#
def prepare_dataloader_for_test(data_path, drop_feature_list=None) -> TimeSeriesDataLoader:
    data_loader = TimeSeriesDataLoader(
        data_path,
        time_series_column_name="Date", time_format="%yyyy-%mm-%dd"
    )

    if drop_feature_list is not None:
        for i in drop_feature_list:
            try:
                data_loader.drop_feature(i)
            except KeyError:
                print("{} is not found, skip".format(i))

    # do one hot encoding if needed
    # data_loader.do_one_hot_encoding_by_col("Hour")
    
    return data_loader
    
#=================================#
# End of methods definition parts #
#=================================#

#----------------------------#
# Preparation of files paths #
#----------------------------#

# experimental title 
TRAIN_EXTEND_NAME = 'TWII_Training'

# model direction
SKLEARN_MODEL_SAVE_DIR = '../../model_store/sklearn/rfc/'
SKLEARN_MODEL_SAVE_NAME = 'sklearn_rfc_'+TRAIN_EXTEND_NAME+'.pickle'

RIVER_MODEL_SAVE_DIR = '../../model_store/river/adarf/'
RIVER_MODEL_SAVE_NAME = 'river_adarf_'+TRAIN_EXTEND_NAME+'.pickle'

LABEL = "LABEL"

# output plot direction
OUTPUT_DIR = '../../output_plot/'

#----------------------------------------------------------#
# Start of Online ML Time Series Training/Testing workflow #
#----------------------------------------------------------#
datapaths_training = "../../data/stock_index_predict/eda_TW50_top30_append.csv"

drop_feature_list = ['DailyReturn', 'Adj Close']

data_loader_for_training = prepare_dataloader_for_test(datapaths_training, drop_feature_list)

training_start_date = '2015-01-01'
training_end_date = '2017-12-31'

model_master_sklearn = SklearnRandomForestClassifierTrainer(
    data_loader=data_loader_for_training,
    model_saving_dir=SKLEARN_MODEL_SAVE_DIR,
    model_name=SKLEARN_MODEL_SAVE_NAME,
    n_tree=100, max_depth=10, criterion='gini',
    training_data_start_time=training_start_date, training_data_end_time=training_end_date,
    label_col=LABEL,
    time_series_col_name='Date', time_format="%yyyy-%mm-%dd"
)

model_master_river = RiverAdaRandomForestClassifier(
    data_loader=data_loader_for_training,
    model_saving_dir=RIVER_MODEL_SAVE_DIR,
    model_name=RIVER_MODEL_SAVE_NAME,
    n_tree=100, max_depth=10, criterion='gini',
    training_data_start_time=training_start_date, training_data_end_time=training_end_date,
    label_col=LABEL,
    time_series_col_name='Date', time_format="%yyyy-%mm-%dd"
)


model_sklearn = model_master_sklearn.get_model()
model_river = model_master_river.get_model()


# # draw sklearn features importance
# feature_names = [f"{i}" for i in data_loader_for_training.get_full_df().head(1)]
# feature_names.remove("Date")
# feature_names.remove("LABEL")
#
#
# importances = model_sklearn.feature_importances_
# # std = np.std([tree.feature_importances_ for tree in model_sklearn.estimators_], axis=0)
# # breakpoint()
# forest_importances = pd.Series(importances, index=feature_names)
# forest_importances.sort_values(ascending=False, inplace=True)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# forest_importances.plot.bar()
# ax.set_title("Feature importances by sklearn RF model (TW stock index)")
# ax.set_ylabel("score")
# fig.tight_layout()
# plt.show()
# breakpoint()


print("Tree basic structure measurements")
# trees = model_river.models
# for i in range(len(trees)):
#     g = trees[i].draw()
#     g.render(OUTPUT_DIR + "tree_inspect/AdaRF_tree{}_Structure_after_training".format(i), format='png')


#====================================#
# End of Model Training/Preparation, #
# Going to do model validation.      #
#====================================#

datapaths_testing = "../../data/stock_index_predict/eda_TW50_top30_append_test_start_from_2018.csv"
data_loader_for_test = prepare_dataloader_for_test(datapaths_testing, drop_feature_list)


#===============================================================#
# Evaluating model by prediction probability distribution check #
#===============================================================#
def run_prediction_proba(pred_proba_list, y_true_list, save_fig_path=None):
    print("checking out prediction probability distribution by ground truth classified")
    if isinstance(pred_proba_list, list):
        print("input probability is list, casting into numpy array")
        pred_np_array = np.array(pred_proba_list)
        ground_true = pd.Series(y_true_list)
    else:
        print("input probability is numpy array")
        pred_np_array = pred_proba_list
        ground_true = y_true_list

    draw_pred_proba = PredictionProbabilityDist(pred_np_array, ground_true)
    print("going to draw probability distribution")
    draw_pred_proba.draw_proba_dist_by_true_false_class_seperated()
    print("finish to draw probability distribution")
    if save_fig_path is not None:
        draw_pred_proba.save_fig(save_fig_path)
        print("Saving prediction probability distribution plot at {} successfully".format(save_fig_path))


#--------------------------------------#
# initializing sklearn model evaluator #
#--------------------------------------#
sklearn_evaluator = SklearnModelEvaluator(
    model_sklearn, data_loader_for_test, LABEL
)
#-----------------------------------------------#
# check out prediction probability distribution #
#-----------------------------------------------#
predict_full_set, y_test_full_set = sklearn_evaluator.predict_proba_true_class_full_set()
run_prediction_proba(predict_full_set, y_test_full_set, OUTPUT_DIR + 'sklearn_pred_proba_plot.pdf')



#--------------------------------------------------------#
# Running Accuracy, recall-rate, and f1 score trend plot #
#--------------------------------------------------------#
sklearn_acc_trend_list = []
sklearn_recall_trend_list = []
sklearn_recall_uncertainty_list = []
sklearn_f1_score_list = []
sklearn_auc_score_list = []
for i_date in data_loader_for_test.get_distinct_date_set_list():
    #===========================================================================#
    # Running prediction probability by date and return daily acc, recall, etc. #
    #===========================================================================#
    pred_result, y_test = sklearn_evaluator.predict_proba_true_class_by_date(i_date)
    acc, recall, recall_uncertainty, f1_s, auc_score = sklearn_evaluator.get_model_score_by_daily_subset(pred_result, y_test, proba_cut=0.5)

    sklearn_acc_trend_list.append(acc * 100)
    sklearn_recall_trend_list.append(recall * 100)
    sklearn_recall_uncertainty_list.append(recall_uncertainty * 100)
    sklearn_f1_score_list.append(f1_s * 100)
    sklearn_auc_score_list.append(auc_score)


#==========================#
# Evaluating RiverML model #
#==========================#
# --------------------------------------#
# initializing riverml model evaluator #
# --------------------------------------#
river_evaluator = RiverModelEvaluator(
    model_river, data_loader_for_test, LABEL
)


#-----------------------------------------------#
# check out prediction probability distribution #
#-----------------------------------------------#
# predict_full_set, y_test_full_set = river_evaluator.predict_proba_true_class_full_set()
#
# roc_curve_display = river_evaluator.roc_curve_displayer(predict_full_set, y_test_full_set, estimator_name="river ml")
# RocCurve(roc_curve_display)
#
# run_prediction_proba(predict_full_set, y_test_full_set, OUTPUT_DIR + 'river_pred_proba_plot.pdf')

river_acc_trend_list = []
river_recall_trend_list = []
river_recall_uncertainty_list = []
river_f1_score_list = []
river_auc_score_list = []

#------------------------------------------------------------------------------#
# accumulating prediction proba and corresponding target for plot distribution #
#------------------------------------------------------------------------------#
predict_set_appending_accumulating = []
y_test_set_appending_accumulating = []

i_count = 0

for i_date in data_loader_for_test.get_distinct_date_set_list():
    #===========================================================================#
    # Running prediction probability by date and return daily acc, recall, etc. #
    #===========================================================================#

    print("running river accumulating training with date:{}".format(str(i_date)))

    pred_result, y_test = river_evaluator.predict_proba_true_class_by_date(i_date, do_online_training=True)

    predict_set_appending_accumulating.extend(pred_result)
    y_test_set_appending_accumulating.extend(y_test)
    print(len(predict_set_appending_accumulating))

    acc, recall, recall_uncertainty, f1_s, auc_score = river_evaluator.get_model_score_by_daily_subset(pred_result, y_test, proba_cut=0.5)

    # roc_curve_display = river_evaluator.roc_curve_displayer(pred_result, y_test, estimator_name="river ml")
    # RocCurve(roc_curve_display)

    river_acc_trend_list.append(acc * 100)
    river_recall_trend_list.append(recall * 100)
    river_recall_uncertainty_list.append(recall_uncertainty * 100)
    river_f1_score_list.append(f1_s * 100)
    river_auc_score_list.append(auc_score)

# print("Tree basic structure measurements")
# trees = model_river.models
# for i in range(len(trees)):
#     g = trees[i].draw()
#     g.render(OUTPUT_DIR + "tree_inspect/AdaRF_tree{}_Structure_after_online_learning".format(i), format='png')

# g = model_river.draw()
# g.render(OUTPUT_DIR + "tree_inspect/AdaRF_tree{}_Structure_after_online_learning".format(0), format='png')


import statistics
sklearn_auc_score_list_smooth = [statistics.mean(sklearn_auc_score_list[i:i+30]) for i in range(len(sklearn_auc_score_list)-30)]
river_auc_score_list_smooth = [statistics.mean(river_auc_score_list[i:i+30]) for i in range(len(river_auc_score_list)-30)]

sklearn_acc_trend_list_smooth = [statistics.mean(sklearn_acc_trend_list[i:i+30]) for i in range(len(sklearn_acc_trend_list)-30)]
river_acc_trend_list_smooth = [statistics.mean(river_acc_trend_list[i:i+30]) for i in range(len(river_acc_trend_list)-30)]

sklearn_f1_score_list_smooth = [statistics.mean(sklearn_f1_score_list[i:i+30]) for i in range(len(sklearn_f1_score_list)-30)]
river_f1_score_list_smooth = [statistics.mean(river_f1_score_list[i:i+30]) for i in range(len(river_f1_score_list)-30)]

sklearn_recall_trend_list_smooth = [statistics.mean(sklearn_recall_trend_list[i:i+30]) for i in range(len(sklearn_recall_trend_list)-30)]
river_recall_trend_list_smooth = [statistics.mean(river_recall_trend_list[i:i+30]) for i in range(len(river_recall_trend_list)-30)]

x_list = data_loader_for_test.get_distinct_date_set_list()

trend_plot_auc = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot_auc.plot_trend(x_list[30:], sklearn_auc_score_list_smooth, label="sklearn AUC")
trend_plot_auc.plot_trend(x_list[30:], river_auc_score_list_smooth, label="river AUC")
trend_plot_auc.save_fig(title="AUC Trend Plot", x_label='date', y_label='AUC', save_fig_path=OUTPUT_DIR+'model_evaluation_trend_plot_auc.pdf')

trend_plot_acc = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot_acc.plot_trend(x_list[30:], sklearn_acc_trend_list_smooth, label="sklearn Accuracy")
trend_plot_acc.plot_trend(x_list[30:], river_acc_trend_list_smooth, label="river Accuracy")
trend_plot_acc.save_fig(title="Accuracy", x_label='date', y_label='Accuracy', save_fig_path=OUTPUT_DIR+'model_evaluation_trend_plot_acc.pdf')


trend_plot_f1_score = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot_f1_score.plot_trend(x_list[30:], sklearn_f1_score_list_smooth, label="sklearn f1 score")
trend_plot_f1_score.plot_trend(x_list[30:], river_f1_score_list_smooth, label="river f1 score")
trend_plot_f1_score.save_fig(title="F1 score trend plot", x_label='date', y_label='f1 score', save_fig_path=OUTPUT_DIR+'model_evaluation_trend_plot_f1_score.pdf')

trend_plot_recall = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot_recall.plot_trend(x_list[30:], sklearn_recall_trend_list_smooth, label="sklearn recall")
trend_plot_recall.plot_trend(x_list[30:], river_recall_trend_list_smooth, label="river recall")
trend_plot_recall.save_fig(title="recall rate trend plot", x_label='date', y_label='recall', save_fig_path=OUTPUT_DIR+'model_evaluation_trend_plot_recall.pdf')
