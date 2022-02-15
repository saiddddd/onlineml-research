from tools.data_loader import TimeSeriesDataLoader

from tools.model_trainer_reader import SklearnRandomForestClassifierTrainer, RiverAdaRandomForestClassifier
from tools.model_evaluator import SklearnModelEvaluator, RiverModelEvaluator

from tools.model_perform_visualization import TrendPlot, PredictionProbabilityDist, RocCurve

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
        time_series_column_name="Date", time_format="%yyyy/%mm/%dd"
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
TRAIN_EXTEND_NAME = 'DOW_30_Training'

# model direction
SKLEARN_MODEL_SAVE_DIR = '../../../model_store/sklearn/rfc/'
SKLEARN_MODEL_SAVE_NAME = 'sklearn_rfc_'+TRAIN_EXTEND_NAME+'.pickle'

RIVER_MODEL_SAVE_DIR = '../../../model_store/river/adarf/'
RIVER_MODEL_SAVE_NAME = 'river_adarf_'+TRAIN_EXTEND_NAME+'.pickle'

LABEL = "LABEL"

# output plot direction
OUTPUT_DIR = '../../../output_plot/'

#----------------------------------------------------------#
# Start of Online ML Time Series Training/Testing workflow #
#----------------------------------------------------------#
datapaths_training = "../../data/stock_index_predict/DOW30.csv"

data_loader_for_training = prepare_dataloader_for_test(datapaths_training)

training_start_date = '2010-01-01'
training_end_date = '2010-12-31'

model_master_sklearn = SklearnRandomForestClassifierTrainer(
    data_loader=data_loader_for_training,
    model_saving_dir=SKLEARN_MODEL_SAVE_DIR,
    model_name=SKLEARN_MODEL_SAVE_NAME,
    n_tree=100, max_depth=20, criterion='gini',
    training_data_start_time=training_start_date, training_data_end_time=training_end_date,
    label_col=LABEL,
    time_series_col_name='Date', time_format="%yyyy/%mm/%dd"
)

model_master_river = RiverAdaRandomForestClassifier(
    data_loader=data_loader_for_training,
    model_saving_dir=RIVER_MODEL_SAVE_DIR,
    model_name=RIVER_MODEL_SAVE_NAME,
    n_tree=100, max_depth=20, criterion='gini',
    training_data_start_time=training_start_date, training_data_end_time=training_end_date,
    label_col=LABEL,
    time_series_col_name='Date', time_format="%yyyy/%mm/%dd"
)


model_sklearn = model_master_sklearn.get_model()
model_river = model_master_river.get_model()

# print("Tree basic structure measurements")
# trees = model_river.models
#
# print(json.dumps(trees[0].model.model_measurements, indent=2, default=str))
# print("Tree structure details inspection")
# print(json.dumps(trees[0].model.model_description(), default=str))
# g = trees[0].model.draw()
# g.render(OUTPUT_DIR + "AdaRF_tree1_Structure_do_parallel", format='png')
    
#====================================#
# End of Model Training/Preparation, #
# Going to do model validation.      #
#====================================#

datapaths_testing = "../../data/stock_index_predict/DOW30.csv"
data_loader_for_test = prepare_dataloader_for_test(datapaths_testing)


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
for i_month_year in data_loader_for_test.get_distinct_month_year_set_list():
    #===========================================================================#
    # Running prediction probability by date and return daily acc, recall, etc. #
    #===========================================================================#
    pred_result, y_test = sklearn_evaluator.predict_proba_true_class_by_month_year(i_month_year)
    acc, recall, recall_uncertainty, f1_s, auc_score = sklearn_evaluator.get_model_score_by_daily_subset(pred_result, y_test, proba_cut=0.4)

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

roc_curve_display = river_evaluator.roc_curve_displayer(predict_full_set, y_test_full_set, estimator_name="river ml")
RocCurve(roc_curve_display)

run_prediction_proba(predict_full_set, y_test_full_set, OUTPUT_DIR + 'river_pred_proba_plot.pdf')

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

for i_month_year in data_loader_for_test.get_distinct_month_year_set_list():
    #===========================================================================#
    # Running prediction probability by date and return daily acc, recall, etc. #
    #===========================================================================#

    print("running river accumulating training with date:{}".format(str(i_month_year)))

    pred_result, y_test = river_evaluator.predict_proba_true_class_by_month_year(i_month_year, do_online_training=True)

    predict_set_appending_accumulating.extend(pred_result)
    y_test_set_appending_accumulating.extend(y_test)
    print(len(predict_set_appending_accumulating))

    """
    for check, to be remove
    """
    # dates_to_draw = [
    #     '2021-01-03', '2021-01-06', '2021-01-09', '2021-01-12', '2021-01-15', '2021-01-18', '2021-01-21', '2021-01-24', '2021-01-27', '2021-01-30'
    # ]
    # if str(i_date) in dates_to_draw:
    #     run_prediction_proba(predict_set_appending_accumulating, y_test_set_appending_accumulating, OUTPUT_DIR + 'river_pred_proba_plot_{}.pdf'.format(str(i_date)))
    #     predict_set_appending_accumulating = []
    #     y_test_set_appending_accumulating = []



    acc, recall, recall_uncertainty, f1_s, auc_score = river_evaluator.get_model_score_by_daily_subset(pred_result, y_test, proba_cut=0.4)

    # roc_curve_display = river_evaluator.roc_curve_displayer(pred_result, y_test, estimator_name="river ml")
    # RocCurve(roc_curve_display)

    river_acc_trend_list.append(acc * 100)
    river_recall_trend_list.append(recall * 100)
    river_recall_uncertainty_list.append(recall_uncertainty * 100)
    river_f1_score_list.append(f1_s * 100)
    river_auc_score_list.append(auc_score)


x_list = data_loader_for_test.get_distinct_month_year_set_list()


trend_plot_auc = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot_auc.plot_trend(x_list, sklearn_auc_score_list, label="sklearn AUC")
trend_plot_auc.plot_trend(x_list, river_auc_score_list, label="river AUC")
trend_plot_auc.save_fig(title="AUC Trend Plot", x_label='date', y_label='AUC', save_fig_path=OUTPUT_DIR+'model_evaluation_trend_plot_auc.pdf')

trend_plot_f1_score = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot_f1_score.plot_trend(x_list, sklearn_f1_score_list, label="sklearn f1 score")
trend_plot_f1_score.plot_trend(x_list, river_f1_score_list, label="river f1 score")
trend_plot_f1_score.save_fig(title="F1 score trend plot", x_label='date', y_label='f1 score', save_fig_path=OUTPUT_DIR+'model_evaluation_trend_plot_f1_score.pdf')

trend_plot_recall = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot_recall.plot_trend(x_list, sklearn_recall_trend_list, label="sklearn recall")
trend_plot_recall.plot_trend(x_list, river_recall_trend_list, label="river recall")
trend_plot_recall.save_fig(title="recall rate trend plot", x_label='date', y_label='recall', save_fig_path=OUTPUT_DIR+'model_evaluation_trend_plot_recall.pdf')
