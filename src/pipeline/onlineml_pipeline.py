import os

from accepts import accepts

from tools.data_loader import TimeSeriesDataLoader

class ExperimentPipeline:

    @staticmethod
    def _preparing_time_series_dataloader(data_path: str, time_series_column_name: str, time_format: str, drop_feature_list=None) -> TimeSeriesDataLoader:
        """
        preparing time series dataloader base on input data path, should provide one and only one date-time column,
        specifying the date-time format is needed.
        :param data_path: str
        :param time_series_column_name: str
        :param time_format: str
        :param drop_feature_list:
        :return:
        """
        data_loader = TimeSeriesDataLoader(
            data_path,
            time_series_column_name=time_series_column_name, time_format=time_format
        )
        if drop_feature_list is not None:
            for i in drop_feature_list:
                try:
                    data_loader.drop_feature(i)
                except KeyError:
                    print("{} is not found, skip".format(i))

        return data_loader


    def __init__(self):

        self._input_training_data_path = None
        self._input_testing_data_path = None

        self._training_dataloader = None
        self._testing_dataloader = None



    @accepts(str, str, str, str, (None, list, str))
    def data_preparation(self, training_data_path: str, testing_data_path: str, time_series_column_name: str, time_format: str, drop_feature_list=None):

        # check the file exist or not
        if (os.path.isfile(training_data_path)) and (os.path.isfile(testing_data_path)):
            self._input_training_data_path = training_data_path
            self._input_testing_data_path = testing_data_path
        else:
            assert FileNotFoundError(
                "Training data path or Testing data path not found!\n" +
                "Training path: {} is found? {}\n".format(training_data_path, os.path.isfile(training_data_path)) +
                "Testing path: {} is found? {}".format(testing_data_path, os.path.isfile(testing_data_path))
            )

        # Preparing dataloader for following experiment workflow

        self._training_dataloader = self._preparing_time_series_dataloader(training_data_path, time_series_column_name, time_format, drop_feature_list)
        self._testing_dataloader = self._preparing_time_series_dataloader(testing_data_path, time_series_column_name, time_format, drop_feature_list)


    def get_training_dataloader(self):
        """
        get training dataloader after data_preparation
        :return: training_dataloader
        """
        return self._training_dataloader

    def get_testing_dataloader(self):
        """
        get testing dataloader after data_preparation
        :return: testing_dataloader
        """
        return self._testing_dataloader



