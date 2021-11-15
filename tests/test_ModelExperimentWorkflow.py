import traceback

from tools.DataPreparation import AirlineDataPreparation, CreditCardPreparation
from task1_model_evaluation.ModelExperimentWorkflow import ModelSklearnWorkflow, ModelRiverOnlineMLWorkflow
import os


def test_model_batch_performance_workflow():
    input_airline = AirlineDataPreparation()
    input_credit_card = CreditCardPreparation()

    try:
        aaa = ModelSklearnWorkflow(input_airline)
        bbb = ModelSklearnWorkflow(input_credit_card)
        assert True
    except Exception as e:
        # e.with_traceback()
        assert False

    input_int = 123
    try:
        ccc = ModelSklearnWorkflow(input_int)
        assert False
    except TypeError as te:
        assert True
        # traceback.print_exception()
    except Exception as e:
        assert False
        # traceback.print_exception()


def test_subset_rows_split_train_dataset():
    aaa = ModelSklearnWorkflow(AirlineDataPreparation())
    features, target = aaa.subset_rows_split_train_dataset(10, 20)
    print(features.shape)
    print(target.shape)

    assert True


def test_batch_pred_test_dataset():

    assert False
