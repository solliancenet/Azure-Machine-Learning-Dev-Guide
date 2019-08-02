# Using automated machine learning for Classification (Code Sample)


## The dataset used in this example

In this classification example we will solve the problem of [Sonar Classification](https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)) where `y` (the class to be predicted) has a cardinality of 2 (binary classification). In this particular case, `X` consists of 60 numerical features (normalized values of sonar readings at various angles) and `y` has two categories - Rock (R) or Mine (M). The problem at hand is to create a classifier that, given the sonar readings, will predict whether an object being detected is a rock or a mine (or, alternatively, whether an object being detected is a mine or not). The case R (not a mine) is coded as 1 and the case M (a mine) is coded as 2.

Note: All code snippets in this section are designed to run in [Azure Notebooks](https://notebooks.azure.com/).

## Prepare your environment

The first step you will perform is installing/updating your Azure Machine Learning service SDK:

```python
!pip install azureml-sdk[automl,notebooks]==1.0.53 azureml-telemetry==1.0.53
```

Note: 1.0.53 is the latest version of the SDK at the time this section was last updated. Replace it with the current latest version number.

Once you have the latest version of the SDK, you will need to do the necessary imports and create a local folder to store various artifacts:

```python
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

import azureml.dataprep as dprep

from azureml.train.automl import AutoMLConfig
from azureml.train.automl.run import AutoMLRun

from azureml.widgets import RunDetails

experiment_name = 'sonar-binary-classifier'
project_folder = './sonar-binary-classifier'

# Create a project_folder if it doesn't exist
if not os.path.exists(project_folder):
    os.makedirs(project_folder)
```

Next, get a hold on the compute resource you will use to train your model:

```python
cpu_cluster = ComputeTarget(workspace=ws, name='aml01')
```
Note: The creation of the `Workspace` variable `ws` is ommited for brewity. Also, we are assuming there already exists an Azure Machine Learning compute resource named `aml01` (which means we are going to submit an AutoML experiment run to a remote compute resource, not the local machine).

## Prepare input data

Once everything is in place, you will start loading, analyzing, and preparing your input data:

```python
data_flow = dprep.read_csv('https://quickstartsws9073123377.blob.core.windows.net/azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/sonar/sonar.all-data.csv', 
                           header=dprep.api.dataflow.PromoteHeadersMode.NONE,
                          infer_column_types=True)
data_flow.get_profile()
X = data_flow.keep_columns(['Column{:d}'.format(x) for x in range(1,61)]).to_pandas_dataframe()
y = data_flow.keep_columns(['Column61']).to_pandas_dataframe()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=111)
```

Notice how we are splitting the original data set into input features (`X`) and output(`y`) - the categorical feature we are looking to predict.

## Create the AutoML configuration

The following elements are needed to create an AutoML configuration:
- A data script which will be used on the remote compute resource to get the data to train the model.
- A `RunConfiguration` which provides details about the creation and initialization of the Python environment on the remote compute resource.
- A set of AutoML experiment settings including the number of iterations, the maximum time the experiment is allowed to run, the primary metric used to rank resulting models,and  the level of logging.

The data script must be saved localy (as it will be referenced by the AutoML configuration) and it has basically the same code you used above to prepare input data:

```python
%%writefile $project_folder/get_data.py

import azureml.dataprep as dprep
from sklearn.model_selection import train_test_split

def get_data():

    data_flow = dprep.read_csv('https://quickstartsws9073123377.blob.core.windows.net/azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/sonar/sonar.all-data.csv', 
                           header=dprep.api.dataflow.PromoteHeadersMode.NONE,
                          infer_column_types=True)

    X = data_flow.keep_columns(['Column{:d}'.format(x) for x in range(1,61)]).to_pandas_dataframe()
    y = data_flow.keep_columns(['Column61']).to_pandas_dataframe()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=111)

    return { "X" : X_train.values, "y" : y_train.values.flatten() }
```

The run configuration includes details about the remote compute resource, the image used to initialize, and the library dependencies that need to be installed:

```python
run_config = RunConfiguration(framework="python")
run_config.target = cpu_cluster
run_config.environment.docker.enabled = True
run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_CPU_IMAGE

dependencies = CondaDependencies.create(
    pip_packages=["scikit-learn", "azureml-sdk[automl]","azureml-dataprep", "azureml-explain-model"])
run_config.environment.python.conda_dependencies = dependencies
```

Finally, you create the settings object and then initialize the `AutoMLConfig` object:

```python
automl_settings = {
    "name": experiment_name,
    "iteration_timeout_minutes": 10,
    "iterations": 10,
    "n_cross_validations": 5,
    "primary_metric": 'accuracy',
    "preprocess": True,
    "max_concurrent_iterations": 10,
    "verbosity": logging.INFO
}

automl_config = AutoMLConfig(task='classification',
                             debug_log='automl_errors.log',
                             path=project_folder,
                             compute_target=cpu_cluster,
                             run_configuration=run_config,
                             data_script=project_folder + "/get_data.py",
                             model_explainability=True,
                             **automl_settings,
                             )
```

## Running the AutoML experiment

Once you have the `AutoMLConfig` object properly initialized, you are ready to submit your experiment. Depending on the various settings you used, the experiment will run for several minutes or more. Using the `show_output=True` options enables you to get updates on the execution.

![Execution progress for an AutoML experiment run](./media/automl-classification-execution-progress.png)

Note how the best value of the specified metric (accuracy in our case) is tracked throughout the entire execution. In this particular example, the `VotingEnsemble` pipeline yielded the top trained model, with an accuracy of 0.8255.

Once the execution is finished you can request detailed information about the results.

![Execution results for an AutoML experiment run](./media/automl-classification-execution-results.png)

Also, you can get all the metrics recorded during the experiment run.

![Execution metrics for an AutoML experiment run](./media/automl-classification-execution-metrics.png)

## Retrieve the best model and use it on test data

Now that you have several trained models ranked based on the metric you specified when configuring the AutoML experiment, you can retrieve the best one and either use it to score immediately (on test data for example) or register and deploy is as a service.

Retrieving the best model and using it to score on test data requires the use of an `AutoMLRun` object:

```python
automl_run = AutoMLRun(experiment, run.parent.id)
best_run, fitted_model = automl_run.get_output()

y_predict = fitted_model.predict(X_test.values)
print(y_predict)
```

You have now successfully configured an AutoML experiment, submitted it to run on a remote compute resource, analyzed the results it produced, retrieved its best trained model, and used this best model to score on test data.

## Next steps

You can learn more about using automated machine learning for Classification by reviewing these links to additional resources:

- [What is automated machine learning?](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml)

Read next: [Using automated machine learning for Regression (Code Sample)](./automl-regression-code-sample.md)