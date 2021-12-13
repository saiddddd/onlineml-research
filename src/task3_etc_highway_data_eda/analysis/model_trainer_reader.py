import abc
from os import path
import pickle

from sklearn.ensemble import RandomForestClassifier
from river.ensemble import AdaptiveRandomForestClassifier

from tools.data_loader import TimeSeriesDataLoader

import abc
import tqdm

class ModelTrainerReader(abc.ABC):

    def __init__(self, training_data_path, model_saving_dir, model_name,
                 n_tree=100, max_depth=10, criterion='gini',
                 time_series_col_name='DateTime', time_format="%yyyy-%mm-%dd %HH:%MM:%SS",
                 label_col="TrafficJam60MinLater",
                 features_to_drop=[]
                 ):

        """
        Model Trainer and reader, preparation of model for this Time Series Dataset study,
        load or create a new model (and training) we want in cases the model is (NOT) exist.

        :param training_data_path:
        :param model_saving_dir:
        :param model_name:
        :param n_tree:
        :param max_depth:
        :param criterion:
        :param time_series_col_name:
        :param time_format:
        :param label_col:
        :param features_to_drop:
        """

        self._model = None
        self._n_tree = n_tree
        self._max_depth = max_depth
        self._criterion = criterion

        self._data_loader = TimeSeriesDataLoader(
            training_data_path,
            time_series_column_name=time_series_col_name,
            time_format=time_format
        )
        self._label = label_col

        #------------------------------------------------#
        # if there have any unwanted features, drop them #
        #------------------------------------------------#
        if len(features_to_drop) > 0:
            for i in features_to_drop:
                self._data_loader.drop_feature(i)

        if not path.isdir(model_saving_dir):
            print("Critical error! model saving directory: {} is not exist, please provide correct path".format(model_saving_dir))
            raise RuntimeError


        self._model_location = model_saving_dir+model_name
        self._is_model_exist = path.isfile(self._model_location)

    @abc.abstractmethod
    def _create_model(self):
        pass

    @abc.abstractmethod
    def _train_model(self):
        pass

    def get_data_loader(self):
        return self._data_loader

    def get_model(self):
        return self._model

    def save_model(self):

        with open(self._model_location, 'wb') as f:
            pickle.dump(self._model, f)
            print("saving Model : {} successfully".format(self._model_location))
    
    
class SklearnRandomForestClassifierTrainer(ModelTrainerReader):
    
    def __init__(self, training_data_path, model_saving_dir, model_name,
                 n_tree=100, max_depth=10, criterion='gini',
                 training_data_start_time='2020-10-01', training_data_end_time='2020-10-10',
                 features_to_drop=[],
                 label_col="TrafficJam60MinLater",
                 time_series_col_name='DateTime', time_format="%yyyy-%mm-%dd %HH:%MM:%SS"
                 ):

        self._training_data_start_time = training_data_start_time
        self._training_data_end_time = training_data_end_time

        super(SklearnRandomForestClassifierTrainer, self).__init__(
            training_data_path, model_saving_dir, model_name,
            n_tree=n_tree, max_depth=max_depth, criterion=criterion,
            time_series_col_name=time_series_col_name, time_format=time_format,
            label_col=label_col,
            features_to_drop=features_to_drop
        )

        def is_parameter_correct(input_p, model_p):
            """
            parameter checker, check loaded model from persist file has
            comparable parameters with user provided
            :param input_p: input parameter
            :param model_p: model internal parameter
            :return: has identical parameter
            :rtype bool:
            """

            if input_p == model_p:
                return True
            else:
                raise ModelParameterException

        if self._is_model_exist:
            '''
            in case of model location exist, check parameters are identical
            '''
            with open(self._model_location, 'rb') as f:
                self._model = pickle.load(f)
                try:
                    model_parameters = self._model.get_params()
                    is_parameter_correct(model_parameters.get("n_estimators"), self._n_tree)
                    is_parameter_correct(model_parameters.get("max_depth"), self._max_depth)
                    is_parameter_correct(model_parameters.get("criterion"), self._criterion)

                except ModelParameterException:
                    '''
                    if model parameters are not identical, recreate model and re-train
                    '''
                    # recreate model
                    del self._model
                    self._create_model()
                    self._train_model()

                except AttributeError:
                    pass

        else:
            '''
            if the provided model is not exist in the disk, create and train
            '''
            self._create_model()
            self._train_model()

    def _create_model(self):

        self._model = RandomForestClassifier(
            n_jobs=-1,
            n_estimators=self._n_tree,
            max_depth=self._max_depth,
            criterion=self._criterion,
            random_state=42,
            verbose=1
        )


    def _train_model(self):

        print("Going to Training Sklearn model, from start date: {} to end date: {}".format(
            self._training_data_start_time,
            self._training_data_end_time
        ))

        sub_df_by_date = self._data_loader.sub_df_by_time_interval(
            self._training_data_start_time,
            self._training_data_end_time
        )

        X = sub_df_by_date
        y = X.pop(self._label)
        X.drop(["DateTime"], axis=1, inplace=True)

        print(X)

        self._model.fit(X, y)


class RiverAdaRandomForestClassifier(ModelTrainerReader):

    def __init__(self, training_data_path, model_saving_dir, model_name,
                 n_tree=100, max_depth=10, criterion='gini',
                 training_data_start_time='2020-10-01', training_data_end_time='2020-10-10',
                 features_to_drop=[],
                 label_col="TrafficJam60MinLater",
                 time_series_col_name='DateTime', time_format="%yyyy-%mm-%dd %HH:%MM:%SS"):


        self._training_data_start_time = training_data_start_time
        self._training_data_end_time = training_data_end_time

        super(RiverAdaRandomForestClassifier, self).__init__(
            training_data_path, model_saving_dir, model_name,
            n_tree=n_tree, max_depth=max_depth, criterion=criterion,
            time_series_col_name=time_series_col_name, time_format=time_format,
            label_col=label_col,
            features_to_drop=features_to_drop
        )

        if self._is_model_exist:
            '''
            in case of model location exist, check parameters are identical
            '''
            with open(self._model_location, 'rb') as f:
                self._model = pickle.load(f)

        else:
            '''
            if the provided model is not exist in the disk, create and train
            '''
            self._create_model()
            self._train_model()


    def _create_model(self):

        self._model = AdaptiveRandomForestClassifier(
            n_models=self._n_tree,
            max_depth=self._max_depth,
            split_criterion=self._criterion,
            grace_period=2000
        )


    def _train_model(self):

        print("Going to Training Sklearn model, from start date: {} to end date: {}".format(
            self._training_data_start_time,
            self._training_data_end_time
        ))

        sub_df_by_date = self._data_loader.sub_df_by_time_interval(
            self._training_data_start_time,
            self._training_data_end_time
        )

        X = sub_df_by_date
        y = X.pop(self._label)
        X.drop(["DateTime"], axis=1, inplace=True)

        for index, raw in tqdm(X.iterrows(), total=X.shape[0]):
            try:
                self._model.learn_one(raw, y[index])
            except:
                print("error happen: X data :{}".format(str(raw)))


    def training_mode_continuously(self, X, y):

        for index, raw in tqdm(X.iterrows(), total=X.shape[0]):
            try:
                self._model.learn_one(raw, y[index])
            except:
                print("error happen: X data :{}".format(str(raw)))


class ModelParameterException(Exception):
    """the exception that model parameter is not identical between user provided and model loader from persist file"""

    def __init__(self, model, msg=None):
        if msg is None:
            msg = "Model parameter is not identical with user provided"
        super(ModelParameterException, self).__init__(msg)

