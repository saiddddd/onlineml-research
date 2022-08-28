from abc import ABC
from sklearn import metrics


class Evaluator(ABC):

    def __init__(self, y_true, y_pred):

        """
        Provided high-level interface to get model performance evaluation
        :param y_true:
        :param y_pred:
        """

        self._y_true = y_true
        self._y_pred = y_pred

    def get_evaluation(self):
        """
        Get evaluation value by corresponding computation,
        :return:
        """

        raise NotImplementedError


class AccuracyEvaluator(Evaluator):

    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def get_evaluation(self) -> float:
        acc = metrics.accuracy_score(self._y_true, self._y_pred)
        return acc


class MeanAbsoluteErrorEvaluator(Evaluator):

    def __init__(self, y_true, y_pred):
        super().__init__(y_true, y_pred)

    def get_evaluation(self) -> float:
        mae = metrics.mean_absolute_error(self._y_true, self._y_pred)
        return mae

