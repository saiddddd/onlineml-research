import abc
from os import path
import pickle

import pandas
from sklearn.ensemble import RandomForestClassifier
from river.ensemble import AdaptiveRandomForestClassifier

from tools.data_loader import TimeSeriesDataLoader

import abc
from tqdm import tqdm

class ModelTrainerReader(abc.ABC):

    def __init__(self, data_loader, model_saving_dir, model_name,
                 n_tree=100, max_depth=10, criterion='gini',
                 training_data_start_time=None, training_data_end_time=None,
                 time_series_col_name='DateTime', time_format="%yyyy-%mm-%dd %HH:%MM:%SS",
                 label_col="TrafficJam60MinLater"
                 ):

        """
        Model Trainer and reader, preparation of model for this Time Series Dataset study,
        load or create a new model (and training) we want in cases the model is (NOT) exist.

        :param data_loader:
        :param model_saving_dir:
        :param model_name:
        :param n_tree:
        :param max_depth:
        :param criterion:
        :param time_series_col_name:
        :param time_format:
        :param label_col:
        """

        self._model = None
        self._n_tree = n_tree
        self._max_depth = max_depth
        self._criterion = criterion

        self._data_loader = data_loader
        # self._data_loader.do_one_hot_encoding_by_col("Hour")
        self._label = label_col

        self._training_data_start_time = training_data_start_time
        self._training_data_end_time = training_data_end_time

        if not path.isdir(model_saving_dir):
            print("Critical error! model saving directory: {} is not exist, please provide correct path".format(model_saving_dir))
            raise RuntimeError


        self._model_location = model_saving_dir+model_name
        self._is_model_exist = path.isfile(self._model_location)

    @abc.abstractmethod
    def _create_model(self):
        """
        internal function to create model if model is not exist!
        or, model parameters set is not identical with user given in run time environment
        :return:
        """
        pass

    @abc.abstractmethod
    def _train_model(self):
        """
        training self._model
        :return:
        """
        pass


    def _prepare_data_to_training(self):

        print("Going to preparing data for training model")

        if (self._training_data_start_time is not None) and (self._training_data_end_time is not None):
            print("Specify date is found! from start date: {} to end date: {}".format(
                self._training_data_start_time,
                self._training_data_end_time
            ))
            training_df = self._data_loader.get_sub_df_by_time_interval(self._training_data_start_time,
                                                                        self._training_data_end_time)
        else:
            print("Specify date is not found! Using full dataset to train model ")
            training_df = self._data_loader.get_full_df()

        X = training_df
        y = X.pop(self._label)
        X.drop(["DateTime"], axis=1, inplace=True)

        return X, y

    def _save_model(self):

        with open(self._model_location, 'wb') as f:
            pickle.dump(self._model, f)
            print("saving Model : {} successfully".format(self._model_location))

    def get_data_loader(self):
        return self._data_loader

    def get_model(self):
        return self._model


    
    
class SklearnRandomForestClassifierTrainer(ModelTrainerReader):
    
    def __init__(self, data_loader, model_saving_dir, model_name,
                 n_tree=100, max_depth=10, criterion='gini',
                 training_data_start_time=None, training_data_end_time=None,
                 label_col="TrafficJam60MinLater",
                 time_series_col_name='DateTime', time_format="%yyyy-%mm-%dd %HH:%MM:%SS"
                 ):

        super(SklearnRandomForestClassifierTrainer, self).__init__(
            data_loader, model_saving_dir, model_name,
            n_tree=n_tree, max_depth=max_depth, criterion=criterion,
            training_data_start_time=training_data_start_time, training_data_end_time=training_data_end_time,
            time_series_col_name=time_series_col_name, time_format=time_format,
            label_col=label_col
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
            print('Model exist')
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
                    print('model hyper parameter issue, recreate and retrain model')
                    # recreate model
                    del self._model
                    self._create_model()
                    self._train_model()
                    self._save_model()

                except AttributeError:
                    pass

        else:
            '''
            if the provided model is not exist in the disk, create and train
            '''
            self._create_model()
            self._train_model()
            self._save_model()

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

        X, y = self._prepare_data_to_training()
        self._model.fit(X, y)


class RiverAdaRandomForestClassifier(ModelTrainerReader):

    def __init__(self, data_loader, model_saving_dir, model_name,
                 n_tree=100, max_depth=10, criterion='gini',
                 training_data_start_time=None, training_data_end_time=None,
                 label_col="TrafficJam60MinLater",
                 time_series_col_name='DateTime', time_format="%yyyy-%mm-%dd %HH:%MM:%SS"):


        super(RiverAdaRandomForestClassifier, self).__init__(
            data_loader, model_saving_dir, model_name,
            n_tree=n_tree, max_depth=max_depth, criterion=criterion,
            training_data_start_time=training_data_start_time, training_data_end_time=training_data_end_time,
            time_series_col_name=time_series_col_name, time_format=time_format,
            label_col=label_col
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
            self._save_model()


    def _create_model(self):

        self._model = AdaptiveRandomForestClassifier(
            n_models=self._n_tree,
            max_depth=self._max_depth,
            split_criterion=self._criterion,
            grace_period=100,
            drift_detector=None
        )


    def _train_model(self):

        X, y = self._prepare_data_to_training()
        for index, raw in tqdm(X.iterrows(), total=X.shape[0]):
            try:
                self._model.learn_one(raw, y[index])
            except:
                print("error happen: X data :{}".format(str(raw)))


class ModelParameterException(Exception):
    """the exception that model parameter is not identical between user provided and model loader from persist file"""

    def __init__(self, msg=None):
        if msg is None:
            msg = "Model parameter is not identical with user provided"
        super(ModelParameterException, self).__init__(msg)

