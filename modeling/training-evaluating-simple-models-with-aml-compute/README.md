# Training and Evaluating a few simple models Using Azure Notebooks and Azure Machine Learning compute (Code Sample)

Azure Machine Learning service provides a code-first experience where you can use the [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py) to start training your models on your local machine and then scale out to use Azure Machine Learning compute target to train better performing, highly accurate machine learning and deep learning models. Azure Machine Learning service supports many of the popular open-source machine learning and deep learning Python packages, such as:

- [Scikit-learn](https://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [Keras](https://keras.io/)

The goal of this article is to show how [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py) can be used to train models locally and on AML compute cluster. We will also look at how to log metrics during training, monitor training runs, and visualize model performance in Azure Machine Learning.

## Training using local compute (of Azure Notebook)

As stated, you can use Azure Machine Learning SDK for Python to train your machine learning models on local compute. In Azure Notebooks, the when your run your training script on local compute, it run on the Azure Notebook compute environment. There are three components needed to run your training script on local compute using Azure Machine Learning: (1) model training script, (2) script run configuration and (3) an experiment in your workspace to run the training job. Let’s look at each of these components next.

### Model training script

You can use any of the popular open-source machine learning and deep learning Python packages to train your model. The following code snippet use the Scikit-learn library for model training.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# Data transformations for numerical and categorical features
numerical = ['...', '...', '...']
categorical = ['...', '...']

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]
    
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', 
                               sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# Setup the data processing and model training pipeline
clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('regressor', GradientBoostingRegressor())])

# Train the model
clf.fit(X_train, y_train)
```

### Script run configuration

The Script Run Configuration defines the training script and the environment needed to run the training job.  There are two main ways to define the required environment: 

- User Managed Environment - When using a user-managed environment, you are responsible for ensuring that all the necessary packages are available in the Python environment you choose to run the script in.

- System Managed Environment - You can ask the Azure Machine Learning service to build a new conda environment for running your script.

The following code snippet shows how to create a system managed environment:

```python
from azureml.core import ScriptRunConfig
from azureml.core.conda_dependencies import CondaDependencies

# Create the script run config that specifies the path to script folder and filename
src = ScriptRunConfig(source_directory='...', script='...')

# Create a system managed environment
system_managed_env = Environment("system-managed-env")
system_managed_env.python.user_managed_dependencies = False

# Specify conda dependencies with scikit-learn
cd = CondaDependencies.create(conda_packages=['scikit-learn'])
system_managed_env.python.conda_dependencies = cd

# Set the system managed environment in the script run config
src.run_config.environment = system_managed_env
```

### Create experiment and start the training job

Finally, you create an experiment in your machine learning workspace, and start the training job. The following code snippet, shows how to the use the script run configuration created above to start a new training job:

```python
from azureml.core import Workspace, Experiment, Run

# Create a new experiment in the machine learning workspace (ws)
experiment_name = '...'
experiment = Experiment(ws, experiment_name)

# Submit the script run config to start the experiment run
run = experiment.submit(src)
```

## Training using AML compute cluster

The section [Introducing AML compute options](../../modeling/feature-engineering-training-evaluation-selection/model-training/aml-compute-options.md) provides details on the various AML compute options and how to create them. In this section we will look at how the submit the above training script to run on Azure Machine Learning Compute cluster.

The key difference, in running the training job on local vs AML compute is to define the appropriate run configuration. As described in the section [Introducing AML Estimators]( ../../modeling/feature-engineering-training-evaluation-selection/model-training/aml-estimators.md), you can also use a specialized estimator that serves as an abstraction to construct run configurations for standard libraries such as Scikit-learn, Keras TensorFlow, PyTorch etc. Here we will look at an example of creating a run configuration using a Docker based environment.

```python
# Create the Dockor environment and specify the conda dependencies
remote_env = Environment("remote-env")
remote_env.docker.enabled = True
remote_env.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])

# Set compute target name that is created in the workspace in the script run config
src.run_config.target = aml_compute_name

# Set the environment in the script run config
src.run_config.environment = remote_env

# Submit the script run config to start the experiment run on the AML compute
run = experiment.submit(src)
```

## Logging during the model training process

The Azure Machine Learning service provides support for monitoring experiment run, logging metrics, saving artifacts, and viewing the results of a run. Also, if you are already using [MLflow]( https://www.mlflow.org/) for managing your machine learning lifecycle, Azure Machine Learning service provides integration with MLFlow for consolidating your logs, metrics, and training artifacts within Azure Machine Learning service. You can refer to the section [Tools to measure model performance](../../modeling/feature-engineering-training-evaluation-selection/model-evaluation/measure-model-performance.md) to learn more on the MLFlow integration with Azure Machine Learning service. In this section, we will look at how Azure Machine Learning SDK for Python allow you to log model metrics and upload artifacts to the run while training an experiment.

The Azure Machine Learning SDK for Python provides support to log a wide variety of data types to the experiment run. This includes, scalar values, lists, row, table, images, upload file or a directory, and also you can tag a run with custom properties. Depending how the metrics are logged, they can be viewed as charts in the run details page. For example, if you log an array of numeric values, or a single numeric value with the same metric name repeatedly, then you can view the metric as a single variable line chart. In another example, if you log a table with two numerical columns – two metrics, or log a row with two columns repeatedly, then you can view the metrics as two variable line chart. 

The following example, shows you how you can log a single numeric value with the same metric name repeatedly and upload files to the experiment run from the model training script.

```python
from azureml.core import Run
import math

# Get the Run from context in which the script is running
run = Run.get_context()

for i in range(len(depths)):
    ...
    y_predict = clf.predict(X_test)
    y_actual = y_test.values.flatten().tolist()
    rmse = math.sqrt(mean_squared_error(y_actual, y_predict))
    run.log('max_depth', depth, 'Maximum depth of the individual regression estimators')
    run.log('rmse', rmse, 'The RMSE score for max_depth: {}'.format(depth))
    print('max_depth: {} RMSE score: {}'.format(depth, rmse))
    ...

# Load files or directory from the machine where the script is running to the run
run.upload_file(destination_path, source_path) # destination, source
```

## Monitoring model training progress

Azure Machine Learning service provides ability to manage your model training runs from within your Python code. For example, in the above section we saw an example of how to start an experiment run. You can also query your run status, run details, cancel run, or mark a run complete. Typically, the model training script is going to generate output, and log metrics to the run as the model is trained. You can monitor your training script output in real-time from within your Python notebooks in two common ways: (1) to call `wait_for_completion(show_output = True)` on the run object, and (2) use the Jupyter notebook widget: `RunDetails`. You can also monitor the training run from the Azure portal by navigating to the `Experiments` section in your `Machine Learning Workspace`, or open the direct link to the run details page in Azure portal that is available from the call `run.get_portal_url()`. In the run details page within Azure portal, you can see properties, metrics, images, and charts that are logged to the experiment run.

The following code examples show you the two main ways to monitor the model training progress from within your notebook.

```python
run.wait_for_completion(show_output = True)
```

   ![Example output from wait_for_completion method on the Run object](../media/model_monitoring_1.png 'Monitoring model training progress')

```python
from azureml.widgets import RunDetails

RunDetails(run).show()
```
   ![Example output from RunDetails Notebook Widget](../media/model_monitoring_2.png 'Monitoring model training progress')

## Visualizing model performance

   ![Example output from RunDetails Notebook Widget showing model performance metrics](../media/model_perf_1.png 'Visualizing model performance')

   ![Snapshot of model performance as seen in Azure portal](../media/model_perf_3.png 'Visualizing model performance')
        
## Next steps

Please see the following additional references on training on local and AML compute custer in Azure Machine Learning service:

- [Azure Machine Learning Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training)
- [Start, monitor, and cancel training runs in Python](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-runs)
- [Log metrics during training runs in Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-track-experiments)
- [Visualize experiment runs and metrics with TensorBoard and Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-monitor-tensorboard)

Read next: [Simplify the process with Automated Machine Learning, a component of Azure Machine Learning service](../simplify-process-with-automated-ml/README.md)
