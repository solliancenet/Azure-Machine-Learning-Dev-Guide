# Overview of real-time inferencing

In a previous chapter, we spent much time talking about [training a machine learning model](../modeling), which is a multi-step process involving data preparation, feature engineering, training, evaluation, and model selection. The model training process can be very compute-intensive, with training times spanning across many hours, days, or weeks depending on the amount of data, type of algorithm used, and other factors. A trained model, on the other hand, is used to make decisions on new data quickly. In other words, it _infers_ things about new data it is given based on its training. Making these decisions on new data on-demand is called real-time inferencing.

In this article, we walk through some sample model deployments to support real-time inferencing, using the tools and services provided by Azure Machine Learning service. If you wish to follow along by executing the sample code, you need to set up your Azure Notebooks and visual interface environments per the instructions within [environment setup](../intro/environment-setup.md).

## Options for logging during web service deployment

When you deploy your model to a web service, the model and its dependencies are packaged within a Docker image that serves as the run environment. Because of this, most of the logging you will do during the web service deployment is Docker-based. When you configure logging and when you need to troubleshoot your deployments, it is important to know the steps that are part of the deployment task:

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

At the end of your web service deployment, when the image build is successful, the image is registered in Azure Container Registry, and the Docker image is deployed to ACI or AKS, you will sometimes encounter errors when the container initializes. As part of container start-up process, the `init()` function in your scoring script is invoked by the system. If there are uncaught exceptions in the `init()` function, you might see **CrashLoopBackOff** error in the error message. You can inspect the Docker logs, as detailed above, but sometimes these logs are not verbose enough to pinpoint the failure. The most common failure that occurs in the `init()` function is when the [Model.get_model_path()](https://docs.microsoft.com/python/api/azureml-core/azureml.core.model.model?view=azure-ml-py#get-model-path-model-name--version-none---workspace-none-) function is called to retrieve the model files within the container. This function will fail if the files cannot be located. To help troubleshoot this and other errors that may occur during container startup, configure the logging level by setting it to DEBUG prior to running a command in the Container shell.

Setting the logging level to DEBUG causes additional information to be captured and logged, which can provide valuable information while troubleshooting.

Example of elevating the logging level prior to calling the `get_model_path()` function:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
from azureml.core.model import Model
print(Model.get_model_path(model_name='my-best-model'))
```

This example prints out the local path (relative to `/var/azureml-app`) in the container where your scoring script expects to find the model file or folder. You can use the local path output to verify if the file or folder is indeed where it is expected to be.

## Deploying to a web service hosted on AKS from an Azure Notebook

Text

## Deploying to a web service hosted on ACI from an Azure Notebook

Text

## Deploying to a web service hosted on AKS or ACI using Azure Machine Learning visual interface

Text

## Deploying to a web service running on FPGAs from an Azure Notebook

Text

## Next steps

- [Reference link]()
- [Reference link]()

Read next: [Related article]()
