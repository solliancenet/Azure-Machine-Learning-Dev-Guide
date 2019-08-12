# Executing an end-to-end DevOps pipeline with Azure Machine Learning and Azure DevOps (Code Sample)

To apply an MLOps-oriented approach to a data science project, you need to address the following issues:

- Implement a CI/CD pipeline with well defined build and deployment pipelines
- Ensure source code versioning plus machine learning model versioning
- Combine DevOps features from Azure DevOps and Azure Machine Learning service

## Prepare your code repository for MLOps

To implement a proper CI/CD pipeline for your data science projects, your source code repository must contain a set of files that are referenced by the Azure DevOps build and release pipelines. The code in these files is used to link Azure DevOps with Azure Machine Learning service.

The following table provides details about these files (`Path` represents the relative path of the file to the repo, you are free to organize the files as you see fit and, thus, change the path):

File Name | Path | Description
--- | --- | ---
`create_aml_cluster.py` | /aml_service | Creates an Azure Machine Learning service compute resource used by the build pipeline to train the machine learning model.
`deploy.py` | /aml_service | Deploys an image containing a trained machine learning model to an Azure Kubernetes Cluster.
`pipelines_master.py` | /aml_service | Creates an Azure Machine Learning pipeline containing two steps - a training step and a model evaluation stept - that is executed in the context of an Experiment. This script is used by the Azure DevOps build pipeline to train, evaluate, and register the machine learning model.
`dependencies.yml` | /environment_setup | Conda environment specification used to prepare Python environments.
`install_requirements.sh` | /environment_setup | Shell file used to set up the Linux environments in which both the Azure DevOps build and release pipelines are run (via the Azure DevOps agents).
`train.py` | /scripts | Implements the training of the machine learning model and its registration with Azure Machine Learning service model registry. Referred by `pipelines_master.py` to define the first step of the pipeline.
`evaluate.py` | /scripts | Implements the evaluation of the trained machine learning model and the creation of the Docker container image used in deployment. The image is created only if the trained machine learning model performs better than the latest model registered.
`score.py` | /scripts | Provides the scoring script used to build the image for a trained model. Referred by `evaluate.py` to build the image.
`azure_pipelines.yml` | / | YML definition of the Azure DevOps build pipeline used to train the machine learning model. Refers `install_requirements.sh`, `dependencies.yml`, `create_aml_cluster.py`, and `pipelines_master.py`.


The [MLOps starter repo](https://github.com/solliancenet/mcw-mlops-starter) contains all these files. You can either download and add the files to your own repo or clone the starter repo directly into your own repo. For the reminder of this section, we will asume your code repo is an Azure DevOps Git repo and you have successfully cloned the `MLOps starter repo` into it.

## Connect Azure DevOps with Azure Machine Learning service

Several steps in both the build and the release Azure DevOps pipelines rely on the capability of successfully accessing the Azure Machine Learning service from Azure DevOps. To enable this, you need to create a new Azure DevOps service connection to the Azure Machine Learning service. 

You will use the Azure Resource Manager service connection template and the Service Principal Authentication at the resource group level:

![Create a new service connection to Azure Machine Learning Service in Azure DevOps](./media/mlops-connect-devops-to-aml.png)

Once you've done this, Azure DevOps and Azure Machine Learning service are connected from the point of view of authentication, with the Azure DevOps pipelines being able to communicate (via the Python SDK) with the Azure Machine Learning service.

## Create and test your Azure DevOps build pipeline

Using the `azure-pipelines.yml` file, you can create a new Azure DevOps build pipeline. If this is your first pipeline, Azure DevOps will recognize the file (provided it is placed in the root of the repo) and offer to create the pipeline from it.

To configure the build pipeline, update the variables at the top of the YML file with the correct values:

```yml

variables:
  resourcegroup: 'Quick-Starts-XXXXX'
  workspace: 'quick-starts-ws-XXXXX'
  experiment: 'quick-starts-mlops'
  aml_compute_target: 'gpucluster'
  model_name: 'compliance-classifier'
  image_name: 'compliance-classifier-image'
```

Notice the code used in the `create_aml_cluster.py` to connect to the compute resource defined by the `aml_compute_target` variable:

```python
try:
    aml_compute = AmlCompute(ws, args.aml_compute_target)
    print("found existing compute target.")
except ComputeTargetException:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 1)    
    aml_compute = ComputeTarget.create(ws, args.aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
```

The approach we are taking in this code is to re-use the compute target between various Azure DevOps build pipeline executions. Also, if it's not already in place when the first build pipeline is executed, we will create it using `ComputeTarget.create()`. An alternate approach can be to provision a separate compute target for each different execution and then, at the end of the execution, delete it. The advantage of this approach is it avoids having a compute cluster that is fully allocated (during a month, for example). Instead, it will allocate Azure Machine Learning service compute resources only as needed, for the limited period of time each Azure DevOps build pipeline is running. The downside of the approach is it makes build pipelines run for longer periods of time (as they always need to wait for the actual provisioning of compute resources to complete).

Once everything is in place, you can test your Azure DevOps build pipeline. For example, you can change your `train.py` file by updating line 146 (change the learning rate specified for the optimizer):

```python
opt = optimizers.RMSprop(lr=0.1) # change this to 0.3
```

If you performed the change directly on the `master` branch, just commit it and the build pipeline will be triggered. If you used a branch, create and approve a new pull request to merge your commit into the `master` branch.

The core part of the build pipeline is the execution of the `pipelines_master.py` which creates in turn an Azure Machine Learning service pipeline consisting of two steps - a training step (implemented by `train.py`) and an evaluation step (implemented by `evaluate.py`):

```python
trainStep = PythonScriptStep(
    name="train",
    script_name="train.py", 
    arguments=["--model_name", args.model_name,
              "--build_number", args.build_number],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=scripts_folder,
    allow_reuse=False
)
print("trainStep created")

evaluate_output = PipelineData('evaluate_output', datastore=def_blob_store)

evaluateStep = PythonScriptStep(
    name="evaluate",
    script_name="evaluate.py", 
    arguments=["--model_name", args.model_name,  
               "--image_name", args.image_name, 
               "--output", evaluate_output],
    outputs=[evaluate_output],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory=scripts_folder,
    allow_reuse=False
)
print("evaluateStep created")

evaluateStep.run_after(trainStep)
steps = [evaluateStep]

pipeline = Pipeline(workspace=ws, steps=steps)
print ("Pipeline is built")
```

Notice the `--build_number` parameter we're sending out to `train.py`. This will be used to tag the trained model (when it's registerd with the Azure Machine Learning service model registry) with the Azure DevOps build number. This will enable you to trace every single trained machine learning model back to the soure code that was used to train it.

Once the pipeline is created, all that's left is to submit it for execution in the context of an Experiment:

```python
pipeline_run = Experiment(ws, experiment_name).submit(pipeline)
print("Pipeline is submitted for execution")

pipeline_run.wait_for_completion(show_output=True)
```

## Training the machine learning model

The `train.py` provided by the [MLOps starter repo](https://github.com/solliancenet/mcw-mlops-starter) trains a simple classifier aimed to identify wheater the free text description of a particular car component is compliant or not. The training data set is loaded from a public url and then prepared to be fed into a neural network.

```python
# Load the car components labeled data
print("Loading car components data...")
data_url = ('https://quickstartsws9073123377.blob.core.windows.net/'
            'azureml-blobstore-0d1c4218-a5f9-418b-bf55-902b65277b85/'
            'quickstarts/connected-car-data/connected-car_components.csv')
car_components_df = pd.read_csv(data_url)
components = car_components_df["text"].tolist()
labels = car_components_df["label"].tolist()
print("Loading car components data completed.")

```

The Keras `Tokenizer` class is used to learn a vocabulary from the entire input data set. The result is then applied to embed all input texts into 100-dimensional index vectors (left padded as needed):

```python
# use the Tokenizer from Keras to "learn" a vocabulary from the entire car components text
print("Tokenizing data...")
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100                                           
training_samples = 90000                                 
validation_samples = 5000    
max_words = 10000      

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(components)
sequences = tokenizer.texts_to_sequences(components)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)
```

Finally, [GloVe](https://nlp.stanford.edu/projects/glove/) vectors are used to embed the elements of the vocabulary (words) into numerical vectors. In turn, these are used to train a neural network that classifies each item as `Compliant` (0) or `Not Compliant` (1). The resulting trained machine learning model is then saved, evaluated and registered with the Azure Machine Learning service model registry:

```python
model.save('./outputs/model.h5')
print("model saved in ./outputs folder")
print("Saving model files completed.")

print('Model evaluation will print the following metrics: ', model.metrics_names)
evaluation_metrics = model.evaluate(x_test, y_test)
print(evaluation_metrics)

run = Run.get_context()
run.log(model.metrics_names[0], evaluation_metrics[0], 'Model test data loss')
run.log(model.metrics_names[1], evaluation_metrics[1], 'Model test data accuracy')

os.chdir("./outputs")

model_description = 'Deep learning model to classify the descriptions of car components as compliant or non-compliant.'
model = Model.register(
    model_path='model.h5',  # this points to a local file
    model_name=args.model_name,  # this is the name the model is registered as
    tags={"type": "classification", "run_id": run.id, "build_number": args.build_number},
    description=model_description,
    workspace=run.experiment.workspace
```

## Evaluating the trained model

The `evaluate.py` script attempts to decide whether the newly trained model is good enough to be a candidate for deployment. To achieve this, the script compares the accuracy of the new model with the accuracy of the latest model available in the model registry. If the accuracy of the new model is better, a new image will be created for it. If not, it will be simply discarded.

```python
current_model_accuracy = -1 # undefined
if current_model != None:
    current_model_run = Run(run.experiment, run_id = current_model.tags.get("run_id"))
    current_model_accuracy = current_model_run.get_metrics().get("acc")
    print('accuracies')
    print(latest_model_accuracy, current_model_accuracy)
    if latest_model_accuracy > current_model_accuracy:
        deploy_model = True
        print('Current model performs better and will be deployed!')
    else:
        print('Current model does NOT perform better and thus will NOT be deployed!')
```

This step is particularly important in CI/CD scenarios where you don't want to get into a situation where a trained model that is significantly worse than its predecessors ends up being published. 

In case the newly trained model passes the accuracy test, the final result of the evaluation step of the Azure Machine Learning service pipeline will be a container image built using the model and the scoring logic provided by `score.py`:

```python
import os
import json
import numpy as np
from keras.models import load_model
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector

def init():
    global model
    global inputs_dc, prediction_dc
    
    try:
        model_name = 'MODEL-NAME' # Placeholder model name
        print('Looking for model path for model: ', model_name)
        model_path = Model.get_model_path(model_name = model_name)
        print('Loading model from: ', model_path)
        model = load_model(model_path)
        print("Model loaded from disk.")
        print(model.summary())

        inputs_dc = ModelDataCollector("model_telemetry", identifier="inputs")
        prediction_dc = ModelDataCollector("model_telemetry", identifier="predictions", feature_names=["prediction"])
    except Exception as e:
        print(e)
        
# note you can pass in multiple rows for scoring
def run(raw_data):
    import time
    try:
        print("Received input: ", raw_data)
        
        inputs = json.loads(raw_data)     
        inputs = np.array(inputs).reshape(-1, 100)
        results = model.predict(inputs).reshape(-1)

        inputs_dc.collect(inputs) #this call is saving our input data into Azure Blob
        prediction_dc.collect(results) #this call is saving our output data into Azure Blob

        print("Prediction created " + time.strftime("%H:%M:%S"))
        
        results = results.tolist()
        return json.dumps(results)
    except Exception as e:
        error = str(e)
        print("ERROR: " + error + " " + time.strftime("%H:%M:%S"))
        return error
```

Notice the use of the `ModelDataCollector` class and its `collect()` method in the scoring script. This is a necessary step in order to be able to enable data collection on the trained model once it is published as a web service.

## Create and test your Azure DevOps release pipeline

To create your release pipeline, you will need to use the build artifacts produced by the build pipeline as an input. In a CI/CD scenario you will set the continuous deployment trigger flag to `Enabled` in Azure DevOps. This will create a release every time a new build is available.

As opposed to the build pipeline, the release pipeline will be created manually in Azure DevOps. Make sure the pipeline is set to run on a `Hosted Ubuntu 1604` build agent and add the following steps:

Step Name | Configuration
--- | ---
`Use Python version` | Set version spec to 3.6.
`Bash` | Specify ```$(System.DefaultWorkingDirectory)/_mlops-quickstart/devops-for-ai/environment_setup/install_requirements.sh``` as the script path and ```$(System.DefaultWorkingDirectory)/_mlops-quickstart/devops-for-ai/environment_setup``` as working directory. `devops-for-ai` is the name of your build artifact (published by the build pipeline) and `_mlops-quickstart` is its alias in the release pipeline.
`Azure CLI` | Use the following inline script to run the `deply.py` script: ```python aml_service/deploy.py --service_name $(service_name) --aks_name $(aks_name) --aks_region $(aks_region) --description $(description)```. Use ```$(System.DefaultWorkingDirectory)/_mlops-quickstart/devops-for-ai``` as working directory.

Finally, make sure you define the following variables on your Azure DevOps release pipeline:

Variable Name | Description
--- | ---
`aks_name` | The name of the Azure Kubernetes Service instance used to deploy the trained machine learning model.
`aks_region` | The region where the AKS cluster will be created (if it's not already available).
`description` | The description of the published web service.
`service_name` | The name of the published web service.

An important note on the `aks_region` variable. Since deploying a Docker container involves copying the image file, it is highly recommended to create the AKS cluster in the same region where the Azure Machine Learning service workspace was originally created. This will prevent cross-region image file movement.

First, the `deploy.py` script will attempt to find a web service with the same name and delete it:

```python
aks_service_name = args.service_name

try:
    service = Webservice(name=aks_service_name, workspace=ws)
    print("Deleting AKS service {}".format(aks_service_name))
    service.delete()
except:
    print("No existing webservice found: ", aks_service_name)
```

Next, it checks whether the AKS cluster is already created (creates a new one if needed):

```python
compute_list = ws.compute_targets
aks_target = None
if aks_name in compute_list:
    aks_target = compute_list[aks_name]
    
if aks_target == None:
    print("No AKS found. Creating new Aks: {} and AKS Webservice: {}".format(aks_name, aks_service_name))
    prov_config = AksCompute.provisioning_configuration(location=aks_region)
    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws, name=aks_name, provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)
    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)
```

Next, it deploys the trained machine learning model as a web service on the AKS cluster:

```python
print("Creating new webservice")
# Create the web service configuration (using defaults)
aks_config = AksWebservice.deploy_configuration(description = args.description, 
                                                tags = {'name': aks_name, 'image_id': image.id})
service = Webservice.deploy_from_image(
    workspace=ws,
    name=aks_service_name,
    image=image,
    deployment_config=aks_config,
    deployment_target=aks_target
)
service.wait_for_deployment(show_output=True)
print(service.state)
```

At the end, the deployment script performs one final step - it calls the newly deployed web service with some test data, just to make sure everything is properly configured and the service is ready for real-time scoring.

You have now created and end-to-end, CI/CD pipeline that implements the most important MLOps concepts and approaches for a data science project.

## Next steps

You can learn more about applying MLOps concepts to data science projects by reviewing these links to additional resources:

- [Train and deploy machine learning models](https://docs.microsoft.com/en-us/azure/devops/pipelines/targets/azure-machine-learning)

Read next: [Conclusion](../conclusion/README.md)