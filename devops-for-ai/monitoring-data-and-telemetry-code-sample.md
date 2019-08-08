# Monitoring a deployed model's collected data and telemetry (Code Sample)

## Enable or disable data collection for a production model

A typical example for a scoring file used when deploying a trained machine learning model looks like this:

```python
#example: scikit-learn and Swagger
import json
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType


def init():
    global model
    # note here "sklearn_regression_model.pkl" is the name of the model registered under
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_path = Model.get_model_path('sklearn_regression_model.pkl')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)


input_sample = np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
output_sample = np.array([3726.995])


@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        # you can return any datatype as long as it is JSON-serializable
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
```

For more details about deploying trained models with Azure Machine Learning read the [Model deployment](../model-deployment/README.md) section.

To add data collection support you need to make serveral changes to the scoring file.

First, add the necessary imports at the begining:

```python
from azureml.monitoring import ModelDataCollector
```
Next, declare the necessary data collection variables in the `init()` function:

```python
global inputs_dc, prediction_dc
inputs_dc = ModelDataCollector("best_model", identifier="inputs", feature_names=["feat1", "feat2", "feat3". "feat4", "feat5", "feat6"])
prediction_dc = ModelDataCollector("best_model", identifier="predictions", feature_names=["prediction1", "prediction2"])
```

Finally, you need to add the the actual collection code to the `run(data)` function:

```python
data = np.array(data)
result = model.predict(data)
inputs_dc.collect(data) #this call is saving our input data into Azure Blob
prediction_dc.collect(result) #this call is saving our input data into Azure Blob
```
Once your scoring file is prepared, you can enable data collection. This can be done either during initial publishing or later, after the model has already been published.

To enable data collection during deployment, you will set the value of the `collect_model_data` parameter to `True`:

```python
aks_config = AksWebservice.deploy_configuration(collect_model_data=True)
```
To enable data collection for an already published model, you will use the `update()` method on `AksWebService`:

```python
from azureml.core.webservice.aks import AksWebservice

aks_service = AksWebservice(ws, 'compliance-classifier-service')
aks_service.update(collect_model_data=True)
```

To disable data collection for an already published model, you will use the `update()` method on `AksWebService`:

```python
from azureml.core.webservice.aks import AksWebservice

aks_service = AksWebservice(ws, 'compliance-classifier-service')
aks_service.update(collect_model_data=False)
```

## Enable or disable Application Insights telemetry for a production model

To enable telemetry during deployment, you will set the value of the `enable_app_insights` parameter to `True`:

```python
aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)
```
To enable data collection for an already published model, you will use the `update()` method on `AksWebService`:

```python
from azureml.core.webservice.aks import AksWebservice

aks_service = AksWebservice(ws, 'compliance-classifier-service')
aks_service.update(enable_app_insights=True)
```

To disable data collection for an already published model, you will use the `update()` method on `AksWebService`:

```python
from azureml.core.webservice.aks import AksWebservice

aks_service = AksWebservice(ws, 'compliance-classifier-service')
aks_service.update(enable_app_insights=False)
```

## Next steps

You can learn more about monitoring a deployed model's collected data and telemetry by reviewing these links to additional resources:

- [Collect data for models in production](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-enable-data-collection)
- [Monitor your Azure Machine Learning models with Application Insights](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-enable-app-insights)

Read next: [Model version management (Code Sample)](./model-version-management-code-sample.md)