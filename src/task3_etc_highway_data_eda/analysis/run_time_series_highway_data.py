from tools.data_loader import TimeSeriesDataLoader

from model_trainer_reader import SklearnRandomForestClassifierTrainer, RiverAdaRandomForestClassifier
from model_evaluator import SklearnModelEvaluator, RiverModelEvaluator

from tools.model_perform_visualization import TrendPlot, PredictionProbabilityDist

import json

DATA_YEAR_MONTH_LIST = [
    '2019_01', '2019_02', '2019_03', '2019_04', '2019_05', '2019_06', #0~6
    '2019_07', '2019_08', '2019_09', '2019_10', '2019_11', '2019_12', #6~12
    '2020_01', '2020_02', '2020_03', '2020_04', '2020_05', '2020_06', #12~18
    '2020_07', '2020_08', '2020_09', '2020_10', '2020_11', '2020_12', #18~24
    '2021_01', '2021_02', '2021_03', '2021_04', '2021_05', '2021_06', #24~30
    '2021_07', '2021_08', '2021_09', '2021_10', '2021_11', '2021_12'  #30~26
]


#=============================================================#
# Definition of functions for following data processing steps #
#=============================================================#
def combine_data_path(year_month) -> str:
    return DATA_HOME_PATH + "highway_traffic_eda_data_ready_for_ml_" + year_month + '.csv'

def prepare_dataloader_for_test(data_path, drop_feature_list: list) -> TimeSeriesDataLoader:
    data_loader = TimeSeriesDataLoader(
        data_path,
        time_series_column_name="DateTime", time_format="%yyyy-%mm-%dd %HH:%MM:%SS"
    )

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

# data paths
DATA_HOME_PATH = "../../../data/highway_etc_traffic/eda_data/"

# experimental title 
TRAIN_EXTEND_NAME = 'highway1_neihuw_2020_10_full'

# model direction
SKLEARN_MODEL_SAVE_DIR = '../../../model_store/sklearn/rfc/'
SKLEARN_MODEL_SAVE_NAME = 'sklearn_rfc_'+TRAIN_EXTEND_NAME+'.pickle'

RIVER_MODEL_SAVE_DIR = '../../../model_store/river/adarf/'
RIVER_MODEL_SAVE_NAME = 'river_adarf_'+TRAIN_EXTEND_NAME+'.pickle'

LABEL = "TrafficJam60MinLater"

# output plot direction
OUTPUT_DIR = '../../../output_plot/'

#----------------------------------------------------------#
# Start of Online ML Time Series Training/Testing workflow #
#----------------------------------------------------------#

datapaths_training = list(map(lambda x : combine_data_path(DATA_YEAR_MONTH_LIST[x]), range(21, 24)))
feature_to_drop = [
    # "DayOfWeek",
    # "Hour",
    "TrafficJam",
    "TrafficJam30MinLater",
    "MeanSpeed",
    "MeanSpeed10MinAgo",
    "MeanSpeed30MinAgo",
    "MeanSpeed60MinAgo",
    "Upstream1MeanSpeed",
    "Upstream2MeanSpeed",
    "Upstream3MeanSpeed",
    "Downstream1MeanSpeed",
    "Downstream2MeanSpeed",
    "Downstream3MeanSpeed"
]

data_loader_for_training = prepare_dataloader_for_test(datapaths_training, feature_to_drop)

training_start_date = '2020-10-01'
training_end_date = '2020-10-31'

model_master_sklearn = SklearnRandomForestClassifierTrainer(
    data_loader=data_loader_for_training,
    model_saving_dir=SKLEARN_MODEL_SAVE_DIR,
    model_name=SKLEARN_MODEL_SAVE_NAME,
    n_tree=300, max_depth=30, criterion='gini',
    training_data_start_time=training_start_date, training_data_end_time=training_end_date
)

model_master_river = RiverAdaRandomForestClassifier(
    data_loader=data_loader_for_training,
    model_saving_dir=RIVER_MODEL_SAVE_DIR,
    model_name=RIVER_MODEL_SAVE_NAME,
    n_tree=100, max_depth=20, criterion='gini',
    training_data_start_time=training_start_date, training_data_end_time=training_end_date
)


model_sklearn = model_master_sklearn.get_model()
model_river = model_master_river.get_model()

model_master_sklearn.save_model()
model_master_river.save_model()

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

datapaths_testing = list(map(lambda x: combine_data_path(DATA_YEAR_MONTH_LIST[x]), range(24, 25)))
data_loader_for_test = prepare_dataloader_for_test(datapaths_testing, feature_to_drop)


#===============================================================#
# Evaluating model by prediction probability distribution check #
#===============================================================#
def run_prediction_proba(pred_proba_list, y_true_list, save_fig_path=None):
    draw_pred_proba = PredictionProbabilityDist(pred_proba_list, y_true_list)
    draw_pred_proba.draw_proba_dist_by_true_false_class_seperated()
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
# --------------------------------------------------------#
sklearn_acc_trend_list = []
sklearn_recall_trend_list = []
sklearn_recall_uncertainty_list = []
sklearn_f1_score_list = []
x_list = data_loader_for_test.get_distinct_date_set_list()

for i_date in data_loader_for_test.get_distinct_date_set_list():
    #===========================================================================#
    # Running prediction probability by date and return daily acc, recall, etc. #
    #===========================================================================#
    pred_result, y_test = sklearn_evaluator.predict_proba_true_class_by_date(i_date)
    acc, recall, recall_uncertainty, f1_s = sklearn_evaluator.get_model_score_by_daily_subset(pred_result, y_test, proba_cut=0.4)

    sklearn_acc_trend_list.append(acc * 100)
    sklearn_recall_trend_list.append(recall * 100)
    sklearn_recall_uncertainty_list.append(recall_uncertainty * 100)
    sklearn_f1_score_list.append(f1_s * 100)
#

# trend_plot = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
# trend_plot.plot_trend(x_list, sklearn_acc_trend_list, label="sklearn accuracy")
# trend_plot.plot_trend_with_error_bar(x_list, sklearn_recall_trend_list, yerr=sklearn_recall_uncertainty_list, markersize=4, capsize=2, label="sklearn recall")
# trend_plot.save_fig(title="Acc Trend Plot", x_label='date', y_label='%', save_fig_path=OUTPUT_DIR+'sklearn_trend_plot.pdf')
#
#



#==========================#
# Evaluating RiverML model #
#==========================#
# --------------------------------------#
# initializing riverml model evaluator #
# --------------------------------------#
river_evaluator = RiverModelEvaluator(
    model_river, data_loader_for_test, LABEL
)

#------------------------------------------------------------------------------#
# accumulating prediction proba and corresponding target for plot distribution #
#------------------------------------------------------------------------------#
predict_set_appending_accumulating = []
y_test_set_appending_accumulating = []
#-----------------------------------------------#
# check out prediction probability distribution #
#-----------------------------------------------#
predict_full_set, y_test_full_set = river_evaluator.predict_proba_true_class_full_set()
run_prediction_proba(predict_full_set, y_test_full_set, OUTPUT_DIR + 'river_pred_proba_plot.pdf')

# predict_set_appending_accumulating.append(predict_full_set)
# y_test_set_appending_accumulating.append(y_test_full_set)

# river_evaluator.run_prediction_probability_distribution_checker(OUTPUT_DIR+'river_pred_proba_plot.pdf')
#
river_acc_trend_list = []
river_recall_trend_list = []
river_recall_uncertainty_list = []
river_f1_score_list = []

i_date_point = 0

for i_date in data_loader_for_test.get_distinct_date_set_list():
    #===========================================================================#
    # Running prediction probability by date and return daily acc, recall, etc. #
    #===========================================================================#

    print("running river accumulating training with date:{}".format(str(i_date)))

    i_date_point += 1

    pred_result, y_test = river_evaluator.predict_proba_true_class_by_date(i_date, do_online_training=True)

    predict_set_appending_accumulating.append(pred_result)
    y_test_set_appending_accumulating.append(y_test)

    if (str(i_date) == '2021-01-15') or (str(i_date) == '2021-01-31'):
        run_prediction_proba(predict_set_appending_accumulating, y_test_set_appending_accumulating, OUTPUT_DIR + 'river_pred_proba_plot_{}.pdf'.format(str(i_date)))

    acc, recall, recall_uncertainty, f1_s = river_evaluator.get_model_score_by_daily_subset(pred_result, y_test, proba_cut=0.4)

    river_acc_trend_list.append(acc * 100)
    river_recall_trend_list.append(recall * 100)
    river_recall_uncertainty_list.append(recall_uncertainty * 100)
    river_f1_score_list.append(f1_s * 100)

    # if (str(i_date) == '2020-12-01') or (str(i_date) == '2021-01-01') or (str(i_date) == '2021-03-01') or (str(i_date) == '2021-05-01') or (str(i_date) == '2021-07-01') :
    #     x_list_incremental = data_loader_for_test.get_distinct_date_set_list()[:i_date_point]
    #     trend_plot = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
    #     trend_plot.plot_trend(x_list, sklearn_acc_trend_list, label="sklearn accuracy")
    #     trend_plot.plot_trend_with_error_bar(x_list, sklearn_recall_trend_list, yerr=sklearn_recall_uncertainty_list, markersize=4, capsize=2, label="sklearn recall")
    #     trend_plot.plot_trend(x_list_incremental, river_acc_trend_list, label="river accuracy")
    #     trend_plot.plot_trend_with_error_bar(x_list_incremental, river_recall_trend_list, yerr=river_recall_uncertainty_list, markersize=4, capsize=2, label="river recall")
    #     trend_plot.save_fig(title="Acc Trend Plot", x_label='date', y_label='%', save_fig_path=OUTPUT_DIR+'river_append_trend_plot_accumulated_date_'+str(i_date)+'.pdf')

x_list = data_loader_for_test.get_distinct_date_set_list()
trend_plot = TrendPlot(figsize_x=14, figsize_y=4, is_time_series=True)
trend_plot.plot_trend(x_list, sklearn_acc_trend_list, label="sklearn accuracy")
trend_plot.plot_trend_with_error_bar(x_list, sklearn_recall_trend_list, yerr=sklearn_recall_uncertainty_list, markersize=4, capsize=2, label="sklearn recall")
trend_plot.plot_trend(x_list, river_acc_trend_list, label="river accuracy")
trend_plot.plot_trend_with_error_bar(x_list, river_recall_trend_list, yerr=river_recall_uncertainty_list, markersize=4, capsize=2, label="river recall")
trend_plot.save_fig(title="Acc Trend Plot", x_label='date', y_label='%', save_fig_path=OUTPUT_DIR+'river_append_trend_plot.pdf')



