import abc

class ModelEvaluator(abc.ABC):

    def __init__(self, model, data_loader_for_testing, label):

        self._model = model
        self._data_loader_for_testing = data_loader_for_testing
        self._label = label

    @abc.abstractmethod
    def run_prediction_probability_distribution_checker(self):
        pass


class SklearnModelEvaluator(ModelEvaluator):
    def __init__(self, model, data_loader_for_testing, label):
        super(SklearnModelEvaluator, self).__init__(model, data_loader_for_testing, label)

    def run_prediction_probability_distribution_checker(self):

        full_testing_dataset = self._data_loader_for_testing.get_full_df()

        X_test = full_testing_dataset
        y_test = X_test.pop(self._label)
        X_test.drop(["DateTime"], axis=1, inplace=True)

        pred_proba_result = self._model.predict_proba(X_test)

        pred_proba_casting_binary = list(map(lambda x: 0 if x < 0.4 else 1, pred_proba_result[:, 1]))

        pred_proba