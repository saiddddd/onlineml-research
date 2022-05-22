from serving.onlineml_model_serving import OnlineMachineLearningModelServing
from serving.onlineml_model_monitor import ModelPerformanceMonitor


if __name__ == '__main__':

    online_ml_server = OnlineMachineLearningModelServing.get_instance()
    model_monitor = ModelPerformanceMonitor.get_instance()

    online_ml_server.run()
    model_monitor.run_dash()

