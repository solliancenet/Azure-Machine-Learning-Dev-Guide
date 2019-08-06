# Reducing model deployment dependencies and improving model inferencing performance with ONNX

The challenges of optimizing the performance of machine learning models in production is extremely hard considering the range of hardware capabilities on different platforms such as cloud vs edge, CPU vs GPU etc. and couple that with the variety of frameworks that available for building and training the machine learning models. To support the idea of providing users the flexibility to train their machine learning models on framework of their choice, and run that model anywhere with optimal performance, is at the core of [Open Neural Network Exchange](https://onnx.ai/) (ONNX) format for representing machine learning models. ONNX is an open format, that is supported by a [community of partners](https://onnx.ai/supported-tools), including Microsoft, who create compatible frameworks and tools.

You can create ONNX models from many frameworks, including PyTorch, Chainer, Microsoft Cognitive Toolkit (CNTK), MXNet, ML.Net, TensorFlow, Keras, SciKit-Learn, and more. ONNX Runtime is optimized for both cloud and edge and works on Linux, Windows, and Mac, and it also integrates with accelerators on different hardware such as TensorRT on NVidia GPUs. [ONNX models can be deployed](https://docs.microsoft.com/azure/machine-learning/service/how-to-build-deploy-onnx#deploy) to the cloud using Azure Machine Learning and ONNX Runtime. They can also be deployed to Windows 10 devices using [Windows ML](https://docs.microsoft.com/windows/ai/). They can even be deployed to other platforms using converters that are available from the ONNX community.

![ONNX flow diagram showing training, converters and deployment](media/onnx_overview.png 'ONNX Flow Diagram')

The interoperability you get with ONNX makes it possible to get great ideas into production faster. With ONNX, data scientists can choose their preferred framework for the job. Similarly, developers can spend less time getting models ready for production, and deploy across the cloud and edge.

- Data scientists: use the framework of their choice to create and train models
- Developers: deploy models cross-platform with minimal integration work

In this article, we will look at how to first convert a model train in Keras with Tensorflow backend to ONNX, and then deploy the ONNX model as a web service hosted on Azure Container Instance (ACI) for making predictions.

## Converting a model to ONNX and deploying to a web service hosted on ACI from an Azure Notebook (Code Sample)

In the [intro](../intro/tools.md) section we learned how to get started with Azure notebooks. As described, Azure Notebooks is a pre-configured environment that is ready to work with Azure Machine Learning service. In this section we will look at code samples that were built in Azure Notebooks.

### Converting deep learning model to ONNX

In this example, we will use a pre-trained Keras model (`model`) with Tensorflow backend. The [ONNXMLTools](https://github.com/onnx/onnxmltools) library provides support for converting models from various machine learning libraries, such as Keras, Tensorflow, scikit-learn, etc. to the ONNX format. The following example, show how to convert a Keras model to ONNX format:

```python
import onnxmltools
import os

# Convert the Keras model to ONNX using target operator set version 7
onnx_model_name = '...'
onnx_model = onnxmltools.convert_keras(model, onnx_model_name, target_opset=7)

# Save the onnx model locally
onnx_model_path = '...'
onnxmltools.utils.save_model(onnx_model, os.path.join(onnx_model_path, onnx_model_name))
```

Next, let's review how you can use the ONNX format to make predictions. It is important to ensure that the input data shape matches the expected input shape by the ONNX inference session.

```python
import onnxruntime
import numpy as np

# Load the ONNX model and observe the expected input shape
onnx_session = onnxruntime.InferenceSession(onnx_model_path, onnx_model_name)

# Confirm that the expected input shape and the shape test data: x_test match
print('Expected input shape: ', onnx_session.get_inputs()[0].shape)
print('Input shape: ', x_test.shape)

# Run ONNX session to make predictions on test data: x_test
x_test = x_test.astype(np.float32)
onnx_session.run(None, {onnx_session.get_inputs()[0].name: x_test})
```

### Deploy deep learning ONNX format model as a web service

#### Create a scoring script

As described in the article [Overview of deployment target options](deployment-target-options.md) the first step to deploy the model to ACI is to register the model in registry hosted in your Azure Machine Learning Service workspace. Following model registration, you need to create the scoring script. Here an example of the scoring script:

```python
%%writefile $score.py
import sys, os, json
import numpy as np
from azureml.core.model import Model
import onnxruntime

def init():
    global model

    try:
        model_name = '...' #model name should be the same as the registered model name
        print('Looking for model path for model: ', model_name)
        model_path = Model.get_model_path(model_name=model_name)
        # Load the ONNX model
        print('Looking for model in: ', model_path)
        model = onnxruntime.InferenceSession(model_path)
        print('Model loaded...')
    except Exception as e:
        print(e)

def run(raw_data):
    try:
        print("Received input: ", raw_data)
        input_data = np.array(json.loads(raw_data)).astype(np.float32)
        # Run an ONNX session to classify the input
        result = model.run(None, {model.get_inputs()[0].name:input_data})
        return result[0]
    except Exception as e:
        error = str(e)
        return error
```

#### Package model

Next, you package the scoring script in a new Docker image. The image has two main components: (1) the environment file that captures all the required dependencies and (2) scoring script file.

```python
# create a Conda dependencies environment file
from azureml.core.conda_dependencies import CondaDependencies
print("Creating conda dependencies file locally...")
conda_packages = ['numpy']
pip_packages = ['azureml-sdk','onnxruntime']
mycondaenv = CondaDependencies.create(conda_packages=conda_packages, pip_packages=pip_packages)

conda_file = '...'
with open(conda_file, 'w') as f:
    f.write(mycondaenv.serialize_to_string())

runtime = 'python'

# create container image configuration
from azureml.core.image import ContainerImage
print("Creating container image configuration...")
image_config = ContainerImage.image_configuration(execution_script = 'score.py',
                                                  runtime = runtime, conda_file = conda_file)

# create the image
from azureml.core import Image
image_name = '...'
image = Image.create(name=image_name, models=[model], image_config=image_config, workspace=ws)

# wait for image creation to finish
image.wait_for_creation(show_output=True)
```

#### Deploy model

For low-scale and CPU-based, ACI is a good option, and can be easily created on the fly as shown below:

```python
from azureml.core.webservice import AciWebservice, Webservice

aci_name = '...'

aci_config = AciWebservice.deploy_configuration(
    cpu_cores = 1,
    memory_gb = 1,
    tags = {'name': aci_name},
    description = '...')

service_name = '...'

aci_service = Webservice.deploy_from_image(deployment_config=aci_config,
                                           image=image,
                                           name=service_name,
                                           workspace=ws)

aci_service.wait_for_deployment(show_output=True)
```

#### Test deployment

Finally, you can test your ACI deployment by consuming your deployed web service endpoint (Scoring URI) over HTTP. You can get the scoring URI by call calling the service object (shown below) or get it from Azure portal as shown in the [Overview of real-time inferencing](real-time-inferencing.md) article.

```python
import requests

url = aci_service.scoring_uri
print('Scoring URI is: {}'.format(url))

# call the webservice over HTTP
headers = {'Content-Type':'application/json'}
response = requests.post(url, json.dumps(x_test.tolist()), headers=headers)

# print the predictions
print(response.text)
```

## Next steps

Please see the following additional references on Azure Machine Learning Visual Interface:

- [ONNX](https://onnx.ai/)
- [ONNX and Azure Machine Learning: Create and accelerate ML models](https://docs.microsoft.com/azure/machine-learning/service/concept-onnx)
- [ONNX on Azure Machine Learning](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx)

Read next: [Creating machine learning pipelines](../creating-machine-learning-pipelines/machine-learning-pipelines.md)
