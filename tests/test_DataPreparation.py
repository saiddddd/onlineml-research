from src.tools.DataPreparation import ArbitraryDataPreparation

def test__data_prepare():

    try:
        load_data = ArbitraryDataPreparation(
            "../data/airline/airline_data.csv", "satisfaction"
        )

        x_df, y_df = load_data.get_pd_df_data()

        print(x_df)
        print(y_df)

        assert True

    except:
        assert False
