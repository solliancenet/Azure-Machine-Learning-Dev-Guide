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

# Set the system managed environment in your script run config
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

[Introducing AML compute options](../../modeling/feature-engineering-training-evaluation-selection/model-training/aml-compute-options.md)

## Logging during the model training process

## Monitoring model training progress

## Visualizing model performance
        
## Next steps

Please see the following additional references on training on local and AML compute custer in Azure Machine Learning service:

- [Azure Machine Learning Notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/training)

Read next: [Simplify the process with Automated Machine Learning, a component of Azure Machine Learning service](../simplify-process-with-automated-ml/README.md)
