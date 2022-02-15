from src.tools.data_loader import TimeSeriesDataLoader

def test_time_series_data_loader():
    try:
        data_loader = TimeSeriesDataLoader("../data/highway/highway_traffic_eda_data_ready_for_ml_2021_01_for_unitest.csv", time_series_column_name="DateTime", time_format="%yyyy-%mm-%dd %HH:%MM:%SS")
        print(data_loader.get_raw_data_tot_num())
        assert True
    except:
        assert False


    first_data = data_loader.get_earliest_data()
    print(first_data)

    last_data =data_loader.get_latest_data()
    print(last_data)

    data_init_time = data_loader.get_earliest_data_time()
    print(data_init_time)

    data_end_time = data_loader.get_latest_data_time()
    print(data_end_time)

    distinct_time_set = data_loader.get_distinct_time_set_list()
    print(len(distinct_time_set))
    print(distinct_time_set)

    while True:
        try:
            next_time = data_loader.get_next_time_from_iteration()
            sub_df = data_loader.get_sub_df_by_time_interval(next_time)
            print(sub_df)
        except StopIteration:
            break

