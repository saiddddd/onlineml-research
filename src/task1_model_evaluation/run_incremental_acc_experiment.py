import numpy as np
import statistics

from river import tree
from river import ensemble
from sklearn.ensemble import RandomForestClassifier

from tools.DataPreparation import AirlineDataPreparation
from ModelExperimentWorkflow import ModelSklearnWorkflow, ModelRiverOnlineMLWorkflow


model_sklearn = RandomForestClassifier(
        n_estimators=50,
        # criterion="entropy",
        max_depth=10,
        random_state=42
    )
model_riverml_HTC = tree.HoeffdingTreeClassifier()
model_riverml_AdaRF = ensemble.AdaptiveRandomForestClassifier(
    n_models=50,
    max_depth=10,
    seed=0,
)


summary_acc_sklearn = []
summary_acc_riverml_HTC = []
summary_acc_riverml_AdaRF = []

for i in range(3):
    ## create model experiment workflow
    exp_flow_sklearn = ModelSklearnWorkflow(AirlineDataPreparation(), random_seed=i)
    exp_flow_riverml_HTC = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=i)
    exp_flow_riverml_AdaRF = ModelRiverOnlineMLWorkflow(AirlineDataPreparation(), random_seed=i)

    exp_flow_sklearn.set_model(model_sklearn)
    exp_flow_riverml_HTC.set_model(model_riverml_HTC)
    exp_flow_riverml_AdaRF.set_model(model_riverml_AdaRF)

    exp_flow_sklearn.train_model_by_arbitrary_full_data(1, 501)
    exp_flow_riverml_HTC.train_model_by_arbitrary_full_data(1, 501)
    exp_flow_riverml_AdaRF.train_model_by_arbitrary_full_data(1, 501)

    exp_flow_sklearn.incremental_prediction(501, 2001)
    exp_flow_riverml_HTC.incremental_prediction(501, 2001)
    exp_flow_riverml_AdaRF.incremental_prediction(501, 2001)
    i_start = 2001
    i_size = 100
    i_step = 100

    x_text_point = []
    acc_result_sklearn = []
    acc_result_riverml_HTC = []
    acc_result_riverml_AdaRF = []

    while(True):

        x_text_point.append(i_start + 0.5 * i_size)

        # sklearn one step operation
        exp_flow_sklearn.incremental_prediction(i_start, i_start + i_size)
        acc, recall = exp_flow_sklearn.incremental_evaluate_model_get_accuracy_recall()
        acc_result_sklearn.append(acc)

        # riverml one step operation
        exp_flow_riverml_HTC.incremental_prediction(i_start, i_start + i_size)
        acc, recall = exp_flow_riverml_HTC.incremental_evaluate_model_get_accuracy_recall()
        acc_result_riverml_HTC.append(acc)
        exp_flow_riverml_HTC.train_model_by_arbitrary_full_data(i_start, i_start + i_size, is_reset_mode=False)

        #riverml adaptive random forest one step operation
        exp_flow_riverml_AdaRF.incremental_prediction(i_start, i_start + i_size)
        acc, recall = exp_flow_riverml_AdaRF.incremental_evaluate_model_get_accuracy_recall()
        acc_result_riverml_AdaRF.append(acc)
        exp_flow_riverml_AdaRF.train_model_by_arbitrary_full_data(i_start, i_start + i_size, is_reset_mode=False)

        i_start+=i_step

        if i_start > 5000:
            break
    summary_acc_sklearn.append(acc_result_sklearn)
    summary_acc_riverml_HTC.append(acc_result_riverml_HTC)
    summary_acc_riverml_AdaRF.append(acc_result_riverml_AdaRF)

summary_acc_sklearn = np.array(summary_acc_sklearn).T.tolist()
summary_acc_riverml_HTC = np.array(summary_acc_riverml_HTC).T.tolist()
summary_acc_riverml_AdaRF = np.array(summary_acc_riverml_AdaRF).T.tolist()

summary_acc_sklearn_mean = [statistics.mean(i) for i in summary_acc_sklearn]
summary_acc_sklearn_error = [statistics.stdev(i) for i in summary_acc_sklearn]

summary_acc_riverml_HTC_mean = [statistics.mean(i) for i in summary_acc_riverml_HTC]
summary_acc_riverml_HTC_error = [statistics.stdev(i) for i in summary_acc_riverml_HTC]

summary_acc_riverml_AdaRF_mean = [statistics.mean(i) for i in summary_acc_riverml_AdaRF]
summary_acc_riverml_AdaRF_error = [statistics.stdev(i) for i in summary_acc_riverml_AdaRF]

x_text_point = [i * 0.0001 for i in x_text_point]
summary_acc_sklearn_mean = [i * 100 for i in summary_acc_sklearn_mean]
summary_acc_riverml_HTC_mean = [i * 100 for i in summary_acc_riverml_HTC_mean]
summary_acc_riverml_AdaRF_mean = [i * 100 for i in summary_acc_riverml_AdaRF_mean]

summary_acc_sklearn_error = [i * 100 for i in summary_acc_sklearn_error]
summary_acc_riverml_HTC_error = [i * 100 for i in summary_acc_riverml_HTC_error]
summary_acc_riverml_AdaRF_error = [i * 100 for i in summary_acc_riverml_AdaRF_error]

# print(x_text_point)
# print(acc_result_sklearn)

from tools.DataVisualization import TrendPlot

aaa = TrendPlot()
aaa.plot_trend_with_error_band(x_text_point, summary_acc_sklearn_mean, y_err=summary_acc_sklearn_error, label='scikit learn RF')
aaa.plot_trend_with_error_band(x_text_point, summary_acc_riverml_HTC_mean, y_err=summary_acc_riverml_HTC_error, label='HT classifier')
aaa.plot_trend_with_error_band(x_text_point, summary_acc_riverml_AdaRF_mean, y_err=summary_acc_riverml_AdaRF_error, label='Adaptive RF')
aaa.save_fig(
    title='Trend plot of Incremental ML model performance',
    x_label='#data accumulated x10000',
    y_label='accuracy (%)'
)