

from abc import abstractmethod, ABC
import pandas as pd
import numpy as np

from time import time

class DataLoader(ABC):

    """
    Abstraction class of data_loader, which is responsible for reading data from file (generally should be csv file)
    """

    def __init__(self, file_path):
        """
        Initialization of DataLoader, which is going to read data from external data file.
        Loaded as pandas dataframe for following usage in ML pipeline
        :param file_path: input data file path. Generally csv files.
        """
        self._raw_df = None

        if isinstance(file_path, str):
            self._raw_df = self._read_data_from_single_file(file_path)
        elif isinstance(file_path, list):
            self._raw_df = self._read_data_from_multiple_files(file_path)


    @staticmethod
    def _read_data_from_single_file(file_path) -> pd.DataFrame:
        """
        Reading data from single file with string of path.
        :param file_path: input file path, should be csv in general case.
        :type file_path: str
        :return: loaded data as pandas dataframe
        :rtype: pd.Dataframe
        """
        df = pd.read_csv(file_path)
        return df

    @staticmethod
    def _read_data_from_multiple_files(file_path_list) -> pd.DataFrame:
        """
        Reading data from multiple files with a list of string.
        :param file_path_list: list of input files path.
        :type file_path_list: list
        :return: loaded data as concat pandas data frame
        :rtype: pd.Dataframe
        """
        concat_dataframe = pd.concat(map(pd.read_csv, file_path_list))
        return concat_dataframe

    def get_raw_df(self):
        return self._raw_df

    def get_raw_data_tot_num(self):
        return len(self._raw_df.index)


class TimeSeriesDataLoader(DataLoader):
    """
    Processing with Time Series Data.
    """

    def __init__(self, file_path,
                 time_series_column_name: str,
                 time_format: str
                 ):
        super().__init__(file_path)
        # specified the time series columns and get the time format
        self._time_series_column_name = time_series_column_name
        self._time_format = time_format

        # extract time series column into datetime format
        self._op_df = self.get_raw_df()
        self._op_df[self._time_series_column_name] = pd.to_datetime(self._op_df[self._time_series_column_name])

        # sorting by time in ascending order
        self.sort_by_time()

        # extract distinct time set
        self._distinct_time_set = sorted(self._op_df[self._time_series_column_name].unique())

        self._distinct_time_set_iterator = iter(list(self._distinct_time_set))

    def sort_by_time(self, ascending_order=True):
        """
        sorting by time
        :return:
        """
        self._op_df.sort_values([self._time_series_column_name], ascending=ascending_order)
        self._op_df.reset_index(inplace=True, drop=True)
        print(self._op_df)
        print("sorting by time in ascending order:{}, successfully".format(ascending_order))


    def get_distinct_time_set_list(self):
        return self._distinct_time_set

    def get_next_time_set_iteration(self):
        next_time = next(self._distinct_time_set_iterator)
        return next_time

    def sub_df_by_time_interval(self, start_time, end_time=None) -> pd.DataFrame:
        """
        Extracting the sub-sector of dataframe within a given time interval,
        inclusive the boundary is True!

        :param start_time: selection start time (include)
        :type start_time: datetime64
        :param end_time: selection end time (include), which can be remained as None
        :type end_time: datetime64
        :return: selected sub dataframe within given time interval
        :rtype: pd.DataFrame
        """
        if end_time is None:
            end_time = start_time

        df = self._op_df[self._op_df[self._time_series_column_name].between(pd.to_datetime(start_time), pd.to_datetime(end_time), inclusive=True)]
        return df

    def reset_time_set_iterator(self):
        self._distinct_time_set_iterator = iter(self._distinct_time_set)

    def get_earliest_data(self) -> pd.DataFrame:
        first_row = self._op_df.iloc[0]
        return first_row

    def get_latest_data(self):
        last_row = self._op_df.iloc[-1]
        return last_row

    def get_earliest_data_time(self) -> time:
        first_row = self.get_earliest_data()
        return first_row[self._time_series_column_name]

    def get_latest_data_time(self) -> time:
        last_row = self.get_latest_data()
        return last_row[self._time_series_column_name]

    def parse_time(self):
        raise NotImplementedError


if __name__ == '__main__':

    from river import tree

    model = tree.HoeffdingTreeClassifier()

    data_loader = TimeSeriesDataLoader(
        "../../data/highway/highway_traffic_eda_data_ready_for_ml_2021_01_for_unitest.csv",
        time_series_column_name="DateTime", time_format="%yyyy-%mm-%dd %HH:%MM:%SS"
    )


    while True:
        try:
            next_time = data_loader.get_next_time_set_iteration()
            sub_df = data_loader.sub_df_by_time_interval(next_time)
            target_y = sub_df.pop("TrafficJam60MinLater")

            print("feeding data with time :{}".format(pd.to_datetime(next_time)))

            for index, row in sub_df.iterrows():
                model.learn_one(row, target_y[index])

        except StopIteration:
            break












