# Model Evaluation introduced

The evaluation of a Machine Learning model is the process through which a set of performance metrics are calculated for an already trained model in order to assess its performance. While these performance metrics play an important role in understanding the performance on the model they were calculated for, they are also critical in the model selection process - a process through which the best model is selected from a number of candidate trained models.

Azure Machine Learning service provides a comprehensive environment to implement Model Evaluation, giving you a centralized place to store and access all model performance data. The collection of performance data can be either performed manually (through the training script itself using the [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/install?view=azure-ml-py)) or automatically in the the case of automated training. There are multiple types of data that can be used to store performance data. Examples include numeric and string scalars, lists, tables, images, and custom files.

Model performance data can be programmatically accessed through the SDK and the Azure Portal can be used to view the raw version of it as well as various graphical representations (like confusion matrices, gain and lift charts, receiver operating characteristics curves, calibration plots, predicted vs. true charts, and histograms of residuals, to name just a few). In addition to the Azure Portal, third party tools like [MLFlow](https://mlflow.org/) and [TensorBoard](https://www.tensorflow.org/tensorboard/) can be integrated.

In this section we will focus on the following:

- [Measuring performance of classification and regression models](./measure-performance-regression-classification.md)
- [Tools to measure model performance](./measure-model-performance.md)
- [Capturing and querying model performance data with AML Experiments](./capture-query-model-performance-with-aml-experiments.md)

## Next steps

You can learn more about Model Evaluation by reviewing these links to additional resources:

- [Start, monitor, and cancel training runs in Python](https://docs.microsoft.com/azure/machine-learning/service/how-to-manage-runs)
- [Log metrics during training runs in Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/service/how-to-track-experiments)
- [Track metrics and deploy models with MLflow and Azure Machine Learning service (Preview)](https://docs.microsoft.com/azure/machine-learning/service/how-to-use-mlflow)
- [Visualize experiment runs and metrics with TensorBoard and Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/service/how-to-monitor-tensorboard)

### Related topics

- [Feature Engineering introduced](./feature-engineering-training-evaluation-selection.md#feature-engineering-introduced)
- [Model Training introduced](./model-training.md)

Read next: [Measuring performance of classification and models](./measure-performance-regression-classification.md)
