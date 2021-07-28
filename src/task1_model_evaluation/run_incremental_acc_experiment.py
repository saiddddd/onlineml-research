import os

import numpy as np
import statistics

from river import tree
from river import ensemble
from sklearn.ensemble import RandomForestClassifier

from tools.DataPreparation import AirlineDataPreparation
from ModelExperimentWorkflow import ModelSklearnWorkflow, ModelRiverOnlineMLWorkflow

print(os.getcwd())

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

#----------------------------------------------------------#
# Starting the loop for randomly sampling training dataset #
#----------------------------------------------------------#
for i in range(10):

    x_text_point = []
    acc_result_sklearn = []
    acc_result_riverml_HTC = []
    acc_result_riverml_AdaRF = []

    ## create model experiment workflow
    exp_flow_sklearn = ModelSklearnWorkflow(AirlineDataPreparation(), random_seed=i)
    exp_flow_riverml_HTC = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=i)
    exp_flow_riverml_AdaRF = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=i)

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
    for exp in exp_flow_dict.values():
        exp['workflow'].train_model_by_arbitrary_split_train_data(1, 501)
        exp['workflow'].incremental_prediction(501, 2001)

    #--------------------------------------------------------------------#
    # Simulating the streaming workflow for following step               #
    # defined the start point and the size of one step,                  #
    # how many #data move forward in the next step(for this start point) #
    #--------------------------------------------------------------------#
    i_start = 2001
    i_size = 100
    i_step = 100
    while(True):

        x_text_point.append(i_start + 0.5 * i_size)

        #--------------------------------------------#
        # To do incremental prediction in this step  #
        # Appending the accuracy result in this step #
        #--------------------------------------------#
        for exp in exp_flow_dict.values():
            exp['workflow'].incremental_prediction(i_start, i_start + i_size)
            acc, recall = exp['workflow'].incremental_evaluate_model_get_accuracy_recall()
            exp['inter_run_acc'].append(acc)

        #--------------------------------------------------------------#
        # For online ml model, to learn more from this coming new data #
        #--------------------------------------------------------------#
        exp_flow_dict.get('river_htc')['workflow'].train_model_by_arbitrary_split_train_data(i_start, i_start + i_size, is_reset_mode=False)
        exp_flow_dict.get('river_adarf')['workflow'].train_model_by_arbitrary_split_train_data(i_start, i_start + i_size, is_reset_mode=False)

        i_start+=i_step

        if i_start > 70000:
            break

    #-----------------------------------------------------------#
    # Appending the inter_run accuracy result into summary list #
    #-----------------------------------------------------------#
    for exp in exp_flow_dict.values():
        exp['summary_acc'].append(exp['inter_run_acc'])

#--------------------------------------#
# Calculating mean value and std error #
#--------------------------------------#
for exp in exp_flow_dict.values():
    exp['summary_acc'] = np.array(exp['summary_acc']).T.tolist()
    exp['summary_acc_mean'] = [statistics.mean(i)*100 for i in exp['summary_acc']]
    exp['summary_acc_error'] = [statistics.stdev(i)*100 for i in exp['summary_acc']]

x_text_point = [i * 0.0001 for i in x_text_point]

from tools.DataVisualization import TrendPlot

aaa = TrendPlot()
aaa.plot_trend_with_error_band(x_text_point, exp_flow_dict.get('sklearn')['summary_acc_mean'], y_err=exp_flow_dict.get('sklearn')['summary_acc_error'], label='scikit learn RF')
aaa.plot_trend_with_error_band(x_text_point, exp_flow_dict.get('river_htc')['summary_acc_mean'], y_err=exp_flow_dict.get('river_htc')['summary_acc_error'], label='HT classifier')
aaa.plot_trend_with_error_band(x_text_point, exp_flow_dict.get('river_adarf')['summary_acc_mean'], y_err=exp_flow_dict.get('river_adarf')['summary_acc_error'], label='Adaptive RF')
aaa.save_fig(
    title='Trend plot of Incremental ML model performance',
    x_label='#data accumulated x10000',
    y_label='accuracy (%)'
)