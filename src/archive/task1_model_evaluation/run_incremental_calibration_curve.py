import os

import numpy as np
import statistics
from tools.DataVisualization import CalibrationPlot

from river import tree
from river import ensemble
from sklearn.ensemble import RandomForestClassifier

from tools.DataPreparation import AirlineDataPreparation
from ModelExperimentWorkflow import ModelSklearnWorkflow, ModelRiverOnlineMLWorkflow

from sklearn.calibration import calibration_curve

#--------------------------#
# Initialization of models #
#--------------------------#
model_sklearn = RandomForestClassifier(
        n_estimators=50,
        criterion="entropy",
        max_depth=10,
        random_state=42
    )
model_riverml_HTC = tree.HoeffdingTreeClassifier()
model_riverml_AdaRF = ensemble.AdaptiveRandomForestClassifier(
    n_models=50,
    max_depth=10,
    seed=0,
)

#------------------------------------------------------#
# create the list for storing accuracy testing results #
#------------------------------------------------------#
summary_acc_sklearn = []
summary_acc_sklearn_mean = []
summary_acc_sklearn_error = []

summary_acc_riverml_HTC = []
summary_acc_riverml_HTC_mean = []
summary_acc_riverml_HTC_error = []

summary_acc_riverml_AdaRF = []
summary_acc_riverml_AdaRF_mean = []
summary_acc_riverml_AdaRF_error = []


x_text_point = []
acc_result_sklearn = []
acc_result_riverml_HTC = []
acc_result_riverml_AdaRF = []

## create model experiment workflow
exp_flow_sklearn = ModelSklearnWorkflow(AirlineDataPreparation(), random_seed=42)
exp_flow_riverml_HTC = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=42)
exp_flow_riverml_AdaRF = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=42)

exp_flow_sklearn.set_model(model_sklearn)
exp_flow_riverml_HTC.set_model(model_riverml_HTC)
exp_flow_riverml_AdaRF.set_model(model_riverml_AdaRF)

#---------------------------------------------------------------#
# Build up the dictionary for storing model experimental object #
# The following work will iterate this dictionary               #
# Taking the individual workflow object to do corresponding job #
#---------------------------------------------------------------#
exp_flow_dict = {'sklearn':
                     {'workflow': exp_flow_sklearn,
                      'inter_run_acc':acc_result_sklearn,
                      'summary_acc':summary_acc_sklearn,
                      'summary_acc_mean':summary_acc_sklearn_mean,
                      'summary_acc_error':summary_acc_sklearn_error},
                 'river_htc':
                     {'workflow':exp_flow_riverml_HTC,
                      'inter_run_acc':acc_result_riverml_HTC,
                      'summary_acc':summary_acc_riverml_HTC,
                      'summary_acc_mean':summary_acc_riverml_HTC_mean,
                      'summary_acc_error':summary_acc_riverml_HTC_error},
                 'river_adarf':
                     {'workflow':exp_flow_riverml_AdaRF,
                      'inter_run_acc':acc_result_riverml_AdaRF,
                      'summary_acc':summary_acc_riverml_AdaRF,
                      'summary_acc_mean':summary_acc_riverml_AdaRF_mean,
                      'summary_acc_error':summary_acc_riverml_AdaRF_mean}
                 }

#---------------------------------#
# first train and initial predict #
#---------------------------------#

cplot = CalibrationPlot()

for name, exp in exp_flow_dict.items():
    exp['workflow'].train_model_by_arbitrary_split_train_data(1, 5000)
    proba, target = exp['workflow'].batch_pred_prob_test_dataset()
    print(name)
    print(proba)
    proba = proba[:, 1]

    fraction_of_positives, mean_predicted_value = calibration_curve(target, proba, n_bins=10)
    cplot.add_calibration_curve(mean_predicted_value, fraction_of_positives, label=name)
    cplot.add_histogram(proba, name)

# exp = exp_flow_dict.get('river_adarf')
# inc_train_data = [0, 50, 1000, 5000, 70000]
# for i in range(len(inc_train_data)):
#
#     if i+1 == len(inc_train_data):
#         break
#     exp['workflow'].train_model_by_arbitrary_split_train_data(inc_train_data[i], inc_train_data[i+1], is_reset_mode=False)
#     proba, target = exp['workflow'].batch_pred_prob_test_dataset()
#     proba = proba[:, 1]
#
#     fraction_of_positives, mean_predicted_value = calibration_curve(target, proba, n_bins=10)
#
#     name = "River ML Adap. RF Trained #{} data".format(inc_train_data[i+1])
#     cplot.add_calibration_curve(mean_predicted_value, fraction_of_positives, label=name)
#     cplot.add_histogram(proba, name)



cplot.save_fig()