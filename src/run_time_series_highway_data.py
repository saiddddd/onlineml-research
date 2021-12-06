from river import tree
from sklearn.ensemble import RandomForestClassifier
from tools.data_loader import TimeSeriesDataLoader


# preparing data
DATA_HOME_PATH = "../data/highway_etc_traffic/eda_data/"

DATA_YEAR_MONTH_LIST = [
    '2020_10', '2020_11', '2020_12', 
    '2021_01', '2021_02', '2021_03', '2021_04', '2021_05', '2021_06', 
    '2021_07', '2021_08', '2021_09', '2021_10', '2021_11', '2021_12'
]


#=============================================================#
# Defination of functions for following data processing steps #
#=============================================================#
def combine_data_path(year_month) -> str:
    return DATA_HOME_PATH + "highway_traffic_eda_data_ready_for_ml_" + year_month + '.csv'


def preparation_data_for_test(data_path) -> TimeSeriesDataLoader:
    data_loader = TimeSeriesDataLoader(
        data_path,
        time_series_column_name="DateTime", time_format="%yyyy-%mm-%dd %HH:%MM:%SS"
    )
    data_loader.drop_feature("TrafficJam")
    data_loader.drop_feature("TrafficJam30MinLater")
    data_loader.drop_feature("MeanSpeed")
    data_loader.drop_feature("MeanSpeed10MinAgo")
    data_loader.drop_feature("MeanSpeed30MinAgo")
    data_loader.drop_feature("MeanSpeed60MinAgo")
    data_loader.drop_feature("Upstream1MeanSpeed")
    data_loader.drop_feature("Upstream2MeanSpeed")
    data_loader.drop_feature("Upstream3MeanSpeed")
    data_loader.drop_feature("Downstream1MeanSpeed")
    data_loader.drop_feature("Downstream2MeanSpeed")
    data_loader.drop_feature("Downstream3MeanSpeed")
    
    full_data = data_loader.get_full_df()

    X = full_data
    y = X.pop("TrafficJam60MinLater")
    X.drop(["DateTime"], axis=1, inplace=True)
    
    return X, y
    
#=================================#
# End of methods defination parts #
#=================================#

#======================#
# Model initialization #
#======================#
model = RandomForestClassifier(
    n_estimators=500,
    criterion="gini",
    max_depth=30,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


datapaths_training = [
    combine_data_path(DATA_YEAR_MONTH_LIST[3])
]
X, y = preparation_data_for_test(datapaths_training)
model.fit(X, y)

datapaths_testing = [
    combine_data_path(DATA_YEAR_MONTH_LIST[3])
]

X_test, y_test = preparation_data_for_test(datapaths_testing)
pred_proba_result = model.predict_proba(X_test)

pred_proba_result_true_class = pred_proba_result[y_test == 1][:, 1]
pred_proba_result_false_class = pred_proba_result[y_test == 0][:, 1]

#====================================#
# Visualization of model performance #
#====================================#
from tools.model_perform_visualization import PredictionProbabilityDist

draw_pred_proba = PredictionProbabilityDist(pred_proba_result, y_test)

draw_pred_proba.draw_proba_dist_by_true_false_class_seperated()
draw_pred_proba.show_plt()
draw_pred_proba.save_fig("../output_plot/highway_pred_proba_distribution_test.pdf")


