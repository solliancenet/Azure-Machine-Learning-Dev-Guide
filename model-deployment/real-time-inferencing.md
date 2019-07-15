# Overview of real-time inferencing

In a previous chapter, we spent much time talking about [training a machine learning model](../modeling), which is a multi-step process involving data preparation, feature engineering, training, evaluation, and model selection. The model training process can be very compute-intensive, with training times spanning across many hours, days, or weeks depending on the amount of data, type of algorithm used, and other factors. A trained model, on the other hand, is used to make decisions on new data quickly. In other words, it _infers_ things about new data it is given based on its training. Making these decisions on new data on-demand is called real-time inferencing.

In this article, we walk through some sample model deployments to support real-time inferencing, using the tools and services provided by Azure Machine Learning service. If you wish to follow along by executing the sample code, you need to set up your Azure Notebooks and visual interface environments per the instructions within [environment setup](../intro/environment-setup.md).

## Options for logging during web service deployment

When you deploy your model to a web service, the model and its dependencies are packaged within a Docker image that serves as the run environment. Because of this, most of the logging you do during the web service deployment is Docker-based. When you configure logging, and when you need to troubleshoot your deployments, it is essential to know the steps that are part of the deployment task:

1. Register the model in the workspace model registry.

2. Build a Docker image, including:

   1. Download the registered model from the registry.
   2. Create a dockerfile, with a Python environment based on the dependencies you specify in the environment yaml file.
   3. Add your model files and the scoring script you supply in the dockerfile.
   4. Build a new Docker image using the dockerfile.
   5. Register the Docker image with the Azure Container Registry associated with the workspace.

3. Deploy the Docker image to Azure Container Instance (ACI) service or to Azure Kubernetes Service (AKS).

4. Start up a new container (or containers) in ACI or AKS.

### Retrieve the image build log URI

If the Docker image build fails during the deployment, you can retrieve detailed logs by obtaining the `image_build_log_uri` value from the image. This URI is a SAS URL that routes to a log file stored in the Azure blob storage account that gets created as part of your workspace. When you run the below commands to retrieve this value, copy and paste it into a browser window to download and view the log file.

```python
# if you already have the image object handy
print(image.image_build_log_uri)

# if you only know the name of the image (note there might be multiple images with the same name but different version number)
print(ws.images['myimg'].image_build_log_uri)

# list logs for all images in the workspace
for name, img in ws.images.items():
    print (img.name, img.version, img.image_build_log_uri)
```

### Inspect the web service Docker log

When you deploy locally or to ACI or AKS, the web service object writes Docker engine log messages that you can print to your console or terminal window:

```python
# if you already have the service object handy
print(service.get_logs())

# if you only know the name of the service (note there might be multiple services with the same name but different version number)
print(ws.webservices['mysvc'].get_logs())
```

### Enable DEBUG logging level within the service container

At the end of your web service deployment, when the image build is successful, the image is registered in Azure Container Registry, and the Docker image is deployed to ACI or AKS, you sometimes encounter errors when the container initializes. As part of the container startup process, the system invokes the `init()` function in your scoring script. If there are uncaught exceptions in the `init()` function, you might see **CrashLoopBackOff** error in the error message. You can inspect the Docker logs, as detailed above, but sometimes these logs are not verbose enough to pinpoint the failure. The most common failure that occurs in the `init()` function is when the [Model.get_model_path()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#get-model-path-model-name--version-none---workspace-none-) function is called to retrieve the model files within the container. This function will fail if it cannot locate the files. To help troubleshoot this and other errors that may occur during container startup, configure the logging level by setting it to DEBUG before running a command in the Container shell.

Setting the logging level to DEBUG causes additional information to be captured and logged, which can provide valuable information while troubleshooting.

Example of elevating the logging level prior to calling the `get_model_path()` function:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
from azureml.core.model import Model
print(Model.get_model_path(model_name='my-best-model'))
```

This example prints out the local path (relative to `/var/azureml-app`) in the container where your scoring script expects to find the model file or folder. You can use the local path output to verify if the file or folder is indeed where it is expected to be.

### View error messages in the Azure portal

In some instances, the Azure portal can be used to view error messages that occur during the deployment process. For instance, if an error occurs during the compute target creation stage of the deployment, you can view the provisioning state and failure output of the compute target by selecting the **Compute** option from the left-hand menu of your Machine Learning service workspace:

![The Compute blade is displayed with the Failed provisioning state value highlighted for the amldev-aks compute target.](media/aml-workspace-compute-failed.png 'Compute blade with failed compute target')

When you select the failed compute target, you can see a detailed output from the failure that can help you troubleshoot the problem. In this example, provisioning a new AKS compute target failed because the core quota limits are exceeded. The error displayed in this example can be resolved by requesting a quota increase for the subscription, by deleting or using an existing compute target, or by reducing the number of requested cores for the compute target you are trying to create.

![The detailed error message is shown and the reason for failure is highlighted: Operation results in exceeding quota limits.](media/aml-workspace-compute-failed-details.png 'Detailed error message for the failed compute target')

## Azure Notebook prerequisites

1. Set up your AML workspace and Azure Notebooks environments per the instructions within [environment setup](../intro/environment-setup.md).
2. Execute the following in a new notebook to configure your environment to use the provided trained ML model. Add each line in its own cell, then execute one by one:

   ```python
   !pip install --upgrade pip
   ```

   ```python
   !pip install --upgrade azureml-sdk[notebooks,explain,automl,contrib]
   ```

   ```python
   !pip install scikit-learn==0.20.3
   ```

   ```python
   !pip install -U scikit-image
   ```

3. Follow the instructions in environment setup to [create a workspace with the SDK](https://github.com/solliancenet/Azure-Machine-Learning-Dev-Guide/blob/master/intro/environment-setup.md#create-a-workspace-with-the-sdk), and run within the Azure Notebook. This allows you to provision all the required Azure resources directly from the notebook without having to use the Azure Portal.

   - If you have a pre-existing AML workspace you want to use, load the workspace with the static `get()` method:

   ```python
   ws = Workspace.get(name="myworkspace", subscription_id='<azure-subscription-id>', resource_group='myresourcegroup')
   ```

### Import required packages

The Azure Machine Learning SDK provides a comprehensive set of a capabilities that you can use directly within a notebook including:

- Creating a **Workspace** that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
- Creating **Experiments** in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
- Creating **Compute** resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU.
- Using **Automated Machine Learning (AutoML)** to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify.
- Packaging a Docker **Image** that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
- Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the **Web Service**.

In Azure Notebooks, all of the libraries needed for Azure Machine Learning are pre-installed. To use them, you just need to import them. Run the following within a new cell in your notebook to do so:

```python
import azureml.core
from azureml.core import Workspace
from azureml.core.webservice import Webservice, AksWebservice
from azureml.core.image import Image
from azureml.core.model import Model

print("Azure ML SDK version:", azureml.core.VERSION)
```

### Download the model that was trained using Automated Machine Learning

Execute the following within a new notebook cell to download the trained ML model for this example:

```python
import urllib.request
import os

model_folder = './automl-model'
model_file_name = 'model.pkl'
model_path = os.path.join(model_folder, model_file_name)

# this is the URL to download a model that was trained using Automated Machine Learning
model_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
             'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
             'quickstarts/automl-model/model.pkl')

# Download the model to your local disk in the model_folder
os.makedirs(model_folder, exist_ok=True)
urllib.request.urlretrieve(model_url, model_path)
```

Now that you have retrieved the model, you must register it. Azure Machine Learning provides a Model Registry that acts like a version-controlled repository for each of your trained models. To version a model, you use the SDK as follows. Run the following within a notebook cell to register the best model with Azure Machine Learning:

```python
# register the model for deployment
model = Model.register(model_path = model_path, # this points to a local file
                       model_name = "nyc-taxi-automl-predictor", # name the model is registered as
                       tags = {'area': "auto", 'type': "regression"},
                       description = "NYC Taxi Fare Predictor",
                       workspace = ws)

print()
print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name,
                                                                                model.description, model.version))
```

### Create a scoring script

Azure Machine Learning SDK gives you control over the logic of the web service, so that you can define how it retrieves the model and how the model is used for scoring. This is an important bit of flexibility. For example, you often have to prepare any input data before sending it to your model for scoring. You can define this data preparation logic (as well as the model loading approach) in the scoring file.

Run the following cell to create a scoring file that will be included in the Docker Image that contains your deployed web service.

**Important** Please update the `model_name` variable in the script below. The model name should be the same as the `Model registered` printed above.

```python
%%writefile scoring_service.py

import json
import numpy as np
import azureml.train.automl as AutoML

columns = ['vendorID', 'passengerCount', 'tripDistance', 'hour_of_day', 'day_of_week', 'day_of_month',
           'month_num', 'normalizeHolidayName', 'isPaidTimeOff', 'snowDepth', 'precipTime',
           'precipDepth', 'temperature']

def init():
    try:
        # One-time initialization of predictive model and scaler
        from azureml.core.model import Model
        from sklearn.externals import joblib
        global model

        model_name = 'nyc-taxi-automl-predictor'
        print('Looking for model path for model: ', model_name)
        model_path = Model.get_model_path(model_name=model_name)
        print('Looking for model in: ', model_path)
        model = joblib.load(model_path)
        print('Model loaded...')

    except Exception as e:
        print('Exception during init: ', str(e))

def run(input_json):
    try:
        inputs = json.loads(input_json)
        # Get the predictions...
        prediction = model.predict(np.array(inputs).reshape(-1, len(columns))).tolist()
        prediction = json.dumps(prediction)
    except Exception as e:
        prediction = str(e)
    return prediction
```

### Package the model

The last step before deployment is to package the model with the scoring script into a new Docker image. This is the image that you deploy to the web service. This step can take several minutes to complete.

```python
# create a Conda dependencies environment file
print("Creating conda dependencies file locally...")
from azureml.core.conda_dependencies import CondaDependencies
conda_packages = ['numpy', 'scikit-learn']
pip_packages = ['azureml-sdk[automl]']
mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)

conda_file = 'automl_dependencies.yml'
with open(conda_file, 'w') as f:
    f.write(mycondaenv.serialize_to_string())

runtime = 'python'

# create container image configuration
print("Creating container image configuration...")
from azureml.core.image import ContainerImage
image_config = ContainerImage.image_configuration(execution_script = 'scoring_service.py',
                                                  runtime = runtime, conda_file = conda_file)

# create the image
image_name = 'nyc-taxi-automl-image'

from azureml.core import Image
image = Image.create(name=image_name, models=[model], image_config=image_config, workspace=ws)

# wait for image creation to finish
image.wait_for_creation(show_output=True)
```

## Deploying to a web service hosted on ACI from an Azure Notebook

When you want to test a model deployment, or if your deployment is very low-scale and CPU-based, [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/) (ACI) is a good option. This fully managed service is the fastest and most straightforward way to deploy an isolated container in Azure, which means that no cluster management or orchestration is required.

Unlike deploying to AKS, you do not need to create ACI containers in advance because they are created on the fly. This means you can go straight to deploying to ACI.

After completing the pre-requisites above, you have used the Azure ML Python SDK to download and register the trained model, create a scoring file, and package these along with dependencies in a Docker image for deployment. To deploy the model to ACI as a web service, execute the following in a new cell within your notebook:

```python
from azureml.core.webservice import AciWebservice, Webservice

aci_name = 'automl-aci-cluster01'

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1,
    memory_gb = 1,
    tags = {'name': aci_name},
    description = 'NYC Taxi Fare Predictor Web Service')

service_name = 'nyc-taxi-automl-service'

aci_service = Webservice.deploy_from_image(deployment_config=aci_config,
                                           image=image,
                                           name=service_name,
                                           workspace=ws)

aci_service.wait_for_deployment(show_output=True)
```

Your cell output should look like the following:

![The ACI deployment cell is displayed with a successful output.](media/aci-deployment-output.png 'ACI deployment output')

Finally, test the deployed model with direct calls on service object:

```python
import json

data1 = [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]

data2 = [[1, 3, 10, 15, 4, 27, 7, 'None', False, 0, 2.0, 1.0, 80],
         [1, 2, 5, 9, 4, 27, 5, 'Memorial Day', True, 0, 0.0, 0.0, 65]]

result = aci_service.run(json.dumps(data1))
print('Predictions for data1')
print(result)

result = aci_service.run(json.dumps(data2))
print('Predictions for data2')
print(result)
```

The cell output of the test should look similar to the following:

![The ACI web service test cell is displayed with a successful output.](media/aci-test-output.png 'ACI test output')

Now, execute the following script in a cell to test consuming your deployed web service endpoint (Scoring URI) over HTTP:

```python
import requests

url = aci_service.scoring_uri
print('ACI Service: {} scoring URI is: {}'.format(service_name, url))
headers = {'Content-Type':'application/json'}

response = requests.post(url, json.dumps(data1), headers=headers)
print('Predictions for data1')
print(response.text)
response = requests.post(url, json.dumps(data2), headers=headers)
print('Predictions for data2')
print(response.text)
```

The cell output of the test should look similar to the following:

![The ACI web service call over HTTP test cell is displayed with a successful output.](media/aci-test-output-http.png 'ACI test output - HTTP')

You can view your ACI deployment in the Azure portal by selecting your AML workspace and selecting **Deployments** in the left-hand menu.

![The ACI deployment is highlighted within the Deployments blade.](media/aml-workspace-deployments-aci.png 'Deployments')

When you select the deployment, you will see details such as the state (whether it is healthy or in a failed state), the service ID, scoring URI, and other details. You also have tabs to view associated models and images.

![The deployment details are displayed.](media/aml-workspace-deployment-details.png 'Deployment details')

## Deploying to a web service hosted on AKS from an Azure Notebook

For large-scale production workloads, [Azure Kubernetes Service](https://docs.microsoft.com/en-us/azure/aks/) (AKS) is a better option than ACI. AKS is a fully managed service that makes it easier to run Kubernetes clusters in Azure. The cluster can be scaled to meet demand, which is an important feature when you have spikes in your workloads. When you deploy your model to AKS as a target, all you are required to do is provision the AKS service in Azure. Once created, Azure Machine Learning service manages the AKS cluster for you, meaning, you do not need to maintain and manage the agent nodes.

Execute the following cell within your notebook to create a new AKS cluster for deployment. Make sure you replace the `aks_name` value with a unique AKS cluster name for your workspace (between 2-16 characters in length). We use the `AksCompute.ClusterPurpose.DEV_TEST` value in the `AksCompute` configuration for this example. In production workloads, remove the DEV_TEST configuration. This script can take up to 20 minutes to execute:

```python
from azureml.core.compute import AksCompute, ComputeTarget

# Use the default configuration, but set the cluster_purpose to DEV_TEST
prov_config = AksCompute.provisioning_configuration(cluster_purpose = AksCompute.ClusterPurpose.DEV_TEST)

aks_name = 'amldev-aks'
# Create the cluster
aks_target = ComputeTarget.create(workspace = ws,
                                    name = aks_name,
                                    provisioning_configuration = prov_config)

# Wait for the create process to complete
aks_target.wait_for_completion(show_output = True)
```

Execute the following code in a new cell within your notebook to deploy the image created in the pre-requisites to the AKS cluster that you created:

```python
from azureml.core.webservice import AksWebservice, Webservice

aks_service_name = 'nyc-taxi-automl-service-aks'

aks_config = AksWebservice.deploy_configuration(
    cpu_cores = 1,
    memory_gb = 1,
    description = 'NYC Taxi Fare Predictor Web Service (AKS)')

aks_service = Webservice.deploy_from_image(deployment_config=aks_config,
                                           deployment_target=aks_target,
                                           image=image,
                                           name=aks_service_name,
                                           workspace=ws)

aks_service.wait_for_deployment(show_output = True)
print(aks_service.state)
print(aks_service.get_logs())
```

Your cell output should look similar to the following if the deployment was successful:

![The cell shows a successful output.](media/aks-deployment-output.png 'AKS deployment output')

Notice that in the deployment script, we are specifying the size of the VM used for the agent nodes by setting the desired number of CPU cores and memory size in gigabytes: `aks_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)`. If you are not certain what values you should use for `cpu_cores` and `memory_gb`, you can optionally profile your model to determine the optimal CPU and memory requirements using either the SDK or CLI. Model profiling results are emitted as a `Run` object. The full details of [the Model Profile schema can be found in the API documentation](https://docs.microsoft.com/python/api/azureml-core/azureml.core.profile.modelprofile?view=azure-ml-py). Learn more about [how to profile your model using the SDK](https://docs.microsoft.com/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#profile-workspace--profile-name--models--inference-config--input-data-). You can also use the profiler before deploying to local or ACI compute targets.

Execute the following script in a cell to test consuming your deployed AKS web service endpoint (Scoring URI) over HTTP. Notice that in this case, we are adding an authorization header. This is because deployments to AKS have authentication enabled by default:

```python
import requests

headers = {'Content-Type':'application/json'}

if aks_service.auth_enabled:
    headers['Authorization'] = 'Bearer '+aks_service.get_keys()[0]

print(headers)

url = aks_service.scoring_uri
print('AKS Service: {} scoring URI is: {}'.format(aks_service_name, url))

response = requests.post(url, json.dumps(data1), headers=headers)
print('Predictions for data1')
print(response.text)
response = requests.post(url, json.dumps(data2), headers=headers)
print('Predictions for data2')
print(response.text)
```

The cell output of the test should look similar to the following:

![The AKS web service call over HTTP test cell is displayed with a successful output.](media/aks-test-output-http.png 'AKS test output - HTTP')

You can view your AKS deployment in the Azure portal by selecting your AML workspace and selecting **Deployments** in the left-hand menu. If you followed the steps in the ACI deployment section, you should see that deployment as well.

![The AKS deployment is highlighted within the Deployments blade.](media/aml-workspace-deployments-aks.png 'Deployments')

When you select the deployment, you will see its details. The screenshot below highlights interesting attributes, including the deployment state, scoring URI, whether authentication is enabled and the keys you can use for the bearer token in the auth header, as well as whether autoscale is enabled for the AKS cluster. You also have tabs to view associated models and images.

![The deployment details are displayed.](media/aml-workspace-deployment-details-aks.png 'AKS deployment details')

## Deploying to a web service running on FPGAs from an Azure Notebook

Text

## Deploying to a web service hosted on AKS or ACI using Azure Machine Learning visual interface

Text

## Next steps

- [Reference link]()
- [Reference link]()

Read next: [Related article]()
