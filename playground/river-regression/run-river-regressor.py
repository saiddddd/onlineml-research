import datetime

from river import tree
from river import evaluate
from river import metrics
from river import datasets
from river import stream
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from matplotlib import pyplot as plt
from matplotlib import dates as mdates

from tqdm import tqdm

def run_in_river_evaluation():

    # df = pd.read_csv("../../data/wind-farm/windfarm_data.csv")
    # datasets = stream.iter_csv(
    #     "../../data/wind-farm/windfarm_data.csv",
    #     target="power",
    #     converters={
    #         "temperature_00": float,
    #         "wind_direction_00": float,
    #         "wind_speed_00": float,
    #         "temperature_08": float,
    #         "wind_direction_08": float,
    #         "wind_speed_08": float,
    #         "temperature_16": float,
    #         "wind_direction_16": float,
    #         "wind_speed_16": float,
    #         "power": float
    #     },
    #     drop=['date']
    # )

    datasets = stream.iter_csv(
        "../../data/stock_index_predict/eda_TW50_top30_append_regression_2010_2017.csv",
        target="LABEL",
        converters={
            "Gold_Close": float,
            "Oil_Close": float,
            "USD_NTD_X_Close": float,
            "DiffToMA10Price": float,
            "DiffToMA10Volume": float,
            "DiffToMA20Price": float,
            "DiffToMA20Volume": float,
            "DiffToMA30Price": float,
            "DiffToMA30Volume": float,
            "DiffToMA90Price": float,
            "DiffToMA90Volume": float,
            "DiffToMA180Price": float,
            "DiffToMA180Volume": float,
            "InterDayDiffRatio": float,
            "OpenToCloseRatio": float,
            "OpenToHighRatio": float,
            "OpenToLowRatio": float,
            "HighToCloseRatio": float,
            "LowToCloseRatio": float,
            "InterDayDiffRatio_1D_ago": float,
            "OpenToCloseRatio_1D_ago": float,
            "OpenToHighRatio_1D_ago": float,
            "OpenToLowRatio_1D_ago": float,
            "HighToCloseRatio_1D_age": float,
            "LowToCloseRatio_1D_ago": float,
            "InterDayDiffRatio_2D_ago": float,
            "OpenToCloseRatio_2D_ago": float,
            "OpenToHighRatio_2D_ago": float,
            "OpenToLowRatio_2D_ago": float,
            "HighToCloseRatio_2D_age": float,
            "LowToCloseRatio_2D_ago": float,
            "Gold_DailyReturn": float,
            "Oil_DailyReturn": float,
            "LABEL": float
        },
        drop=['Date']
    )

    model = tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=50,
        leaf_prediction='adaptive',
        model_selector_decay=0.3,
        seed=0
    )

    metric = metrics.MAE()

    evaluate.progressive_val_score(datasets, model, metric, print_every=100)


def prepare_training_data():

    training_df = pd.read_csv("../../data/stock_index_predict/eda_TW50_top30_append_regression_2010_2017.csv")
    training_y = training_df.pop("LABEL")
    training_df.drop(columns=['Date'], inplace=True)

    return training_df, training_y


def run_model_training():

    df, y = prepare_training_data()
    print(df.head(5))

    sklearn_model = DecisionTreeRegressor()

    sklearn_model.fit(df, y)

    river_model = tree.HoeffdingAdaptiveTreeRegressor(
        grace_period=50,
        leaf_prediction='adaptive',
        model_selector_decay=0.3,
        seed=0
    )

    c_index = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if c_index > 2000:
            break
        x = row.to_dict()
        river_model.learn_one(x, y[index])
        c_index += 1

    return sklearn_model, river_model

def run_river_model_prediction(model):

    model = model
    pred_result = []
    df_test = pd.read_csv("../../data/stock_index_predict/eda_TW50_top30_append_regression_test_2018.csv")
    df_test_go_training = df_test.drop(columns=['Date'])
    y = df_test_go_training.pop("LABEL")

    for index, row in tqdm(df_test_go_training.iterrows(), total=df_test_go_training.shape[0]):

        x = row.to_dict()
        pred_temp = model.predict_one(x)

        if abs(pred_temp) > 100:
            pred_temp = 0
        pred_result.append(pred_temp)

        model.learn_one(x, y[index])

    df_test['PREDICTION'] = pred_result

    print(df_test.columns)
    print(df_test.dtypes)

    check_df = df_test[['Date', 'LABEL', 'PREDICTION']]

    print(check_df)

    return check_df


def run_sklearn_model_prediction(model):

    model = model

    df_test = pd.read_csv("../../data/stock_index_predict/eda_TW50_top30_append_regression_test_2018.csv")
    df_test_go_training = df_test.drop(columns=['Date'])
    y = df_test_go_training.pop("LABEL")

    pred_result = model.predict(df_test_go_training)

    df_test['PREDICTION'] = pred_result

    print(df_test.columns)
    print(df_test.dtypes)

    check_df = df_test[['Date', 'LABEL', 'PREDICTION']]

    print(check_df)

    return check_df


def draw_stock_pred(x, y):

    df_test = pd.read_csv("../../data/stock_index_predict/eda_TW50_top30_append_regression_test_2018.csv")

    df_groupby_date = df_test.groupby(['Date'])["LABEL"].mean()
    ans = df_groupby_date.to_list()
    # print(ans)

    x_axis = df_test['Date'].unique().tolist()
    # print(x_axis)

    x_axis = [ datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in x_axis ]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=120))
    plt.plot(x_axis, ans)
    plt.gcf().autofmt_xdate()
    plt.show()



if __name__ == "__main__":
    # run_in_river_evaluation()
    sk_model, model = run_model_training()

    check_df_river = run_river_model_prediction(model)
    check_df_sklearn = run_sklearn_model_prediction(sk_model)
    print(check_df_sklearn)

    x_list = check_df_river['Date'].unique().tolist()
    x_list = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in x_list]

    pred_list_river = check_df_river.groupby(['Date'])['PREDICTION'].mean()
    pred_list_sklearn = check_df_sklearn.groupby(['Date'])['PREDICTION'].mean()
    y_true = check_df_river.groupby(['Date'])['LABEL'].mean()

    print("check prediction")

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=120))
    plt.gca().set_ylim([-5, 13])

    plt.plot(x_list, y_true, label="true")
    plt.plot(x_list, pred_list_sklearn, label="sklearn")
    plt.plot(x_list, pred_list_river, label="river")
    plt.legend(loc='best')
    plt.ylabel("Daily Return %")
    plt.gcf().autofmt_xdate()
    plt.show()


    #
    # mae calculating

    from sklearn.metrics import mean_absolute_error

    mae_sklearn = []
    mae_river = []

    for i in range(len(x_list)-30):
        mae_sklearn.append(mean_absolute_error(y_true[i:i+30], pred_list_sklearn[i:i+30]))
        mae_river.append(mean_absolute_error(y_true[i:i+30], pred_list_river[i:i+30]))

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=120))
    # plt.gca().set_ylim([-5, 13])

    # plt.plot(x_list[30:], y_true, label="true")
    plt.plot(x_list[15:-15], mae_sklearn, label="sklearn")
    plt.plot(x_list[15:-15], mae_river, label="river")
    plt.ylabel("MAE")
    plt.legend(loc='best')
    plt.gcf().autofmt_xdate()
    plt.show()



    # draw_stock_pred()
