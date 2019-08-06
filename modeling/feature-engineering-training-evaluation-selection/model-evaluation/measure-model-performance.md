# Tools to measure model performance

The following categories of tools are available for measuring model performance:

- Programatic tools - the [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/install?view=azure-ml-py) and the [Azure Command-Line Interface](https://docs.microsoft.com/cli/azure/?view=azure-cli-latest) (CLI) (plus the [CLI extension for Azure Machine Learning service](https://docs.microsoft.com/azure/machine-learning/service/reference-azure-machine-learning-cli))
- Automated tools - [Azure Machine Learning service Automated ML](../../simplify-process-with-automated-ml/README.md) (AutoML)
- Third party tools - [MLFlow](https://mlflow.org/) and [TensorBoard](https://www.tensorflow.org/tensorboard/)

## Azure Machine Learning SDK for Pyhton and Azure CLI

The SDK and the CLI can be used for the following tasks related to monitoring, organizing, and managing experiments and runs:

- Monitor run performance
- Cancel or fail runs
- Create child runs
- Tag and search for runs

Training code based on the SDK can log the following types of values:

- Scalar values (string or numerical) - logging the same metric multiple times will result in a vector
- Vector (list)
- Row
- Table
- Image
- File
- Tag

These values are typically generated during the model training process and they include telemetry related to model performance. For a detailed discussion about using the SDK to measure model performance, read the [Capturing and querying model performance data with AML Experiments](./capture-query-model-performance-with-aml-experiments.md) section.

## Azure Machine Learning service AutoML

AutoML is an Azure Machine Learning feature that enables you to train and tune a Classification, Regresion, or Time Series Forecasting model using a specific target metric. To achieve this, AutoML iterates through several algorithms and parameterizations producing on each iteration a model and a training score. When it comes to measuring model performance, AutoML provides out-of-the box support for a significant number of metrics and graphical representations.

The [Measuring performance of classification and regression models](./measure-performance-regression-classification.md) section contains a detailed discussion about the performance metrics supported by AutoML. For an exhaustive list of metrics logged by AutomML, read [Evaluate training accuracy in automated ML with metrics](https://docs.microsoft.com/azure/machine-learning/service/how-to-understand-accuracy-metrics).

## MLFlow

[MLFlow](https://mlflow.org/) is a library designed to manage the life cycle of machine learning experiments, originating from the [Databricks ecosystem](https://databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html). MLFlow tracking is a component from MLFlow that provides functions for logging and tracking run metrics and model artifacts.

The following diagram display the conceptual approach of MLFlow:

![](media/model-performance-mlflow.png)

To configure the integration between MLFlow and Azure Machine Learning service, you will use the `get_ml_flow_tracking_uri` method from the `Workspace` object as show in the following example:

```python
import mlflow
from azureml.core import Workspace

ws = Workspace.from_config()

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

experiment_name = 'experiment_with_mlflow'
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_metric('alpha', 0.03)
```

## TensorBoard

The [TensorBoard package](https://docs.microsoft.com/python/api/azureml-tensorboard/?view=azure-ml-py) that is part of the main Azure Machine Learning service SDK for Python enables you to view your runs and metrics in [TensorBoard](https://www.tensorflow.org/tensorboard/). TensorBoard is a suite of web applications for inspecting and understanding your TensorFlow runs and graphs.

There are two options for integration with the TensorBoard environment:

- For experiments where TensorBoard-compatible log files are generated (e.g. experiments driven by TensorFlow/Keras, PyTorch, or Chainer estimators), TensorBoard can be launched directly from the run history available in Azure Machine Learning. For more details about these estimators read the [Introducing AML Estimators](../model-training/aml-estimators.md) section.
- For other experiments (e.g. driven by Scikit-learn estimators), the `export_to_tensorboard` method can be used to export the run history of an experiment in a TensorBoard-compatible log format.

## Next steps

You can learn more about the tools available to measure model performance by reviewing these links to additional resources:

- [Log metrics during training runs in Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/service/how-to-track-experiments)
- [Evaluate training accuracy in automated ML with metrics](https://docs.microsoft.com/azure/machine-learning/service/how-to-understand-accuracy-metrics)
- [Track metrics and deploy models with MLflow and Azure Machine Learning service (Preview)](https://docs.microsoft.com/azure/machine-learning/service/how-to-use-mlflow)
- [Visualize experiment runs and metrics with TensorBoard and Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/service/how-to-monitor-tensorboard)

Read next: [Capturing and querying model performance data with AML Experiments](./capture-query-model-performance-with-aml-experiments.md)

