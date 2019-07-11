# Overview of machine learning pipelines using the Azure Machine Learning SDK

## What are machine learning pipelines?

Machine learning pipelines are cyclical and iterative in nature that facilitate both continuous improvement of model performance and deploying and making inferences on the best performing model to date. The pipelines comprise of distinct steps, for example, data preparation, model training, and batch predictions. Often data scientists, data engineers and IT professionals need to collaborate on building robust, scalable, and reusable machine learning pipelines.

The following diagram shows an example pipeline:

![azure machine learning piplines](./media/pipelines.png)

The [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/?view=azure-ml-py) allows you to create ML pipelines, and also submit and track individual pipeline runs. You can build reusable pipelines that optimize your specific workflows and allows you to focus on your expertise, for example machine learning, instead of the infrastructure to build and manage the pipelines.

The purpose of this article to show how to build an example machine learning pipeline work flow, that includes repeatable data preparation, model training and batch predictions using the Azure Machine Learning SDK for Python within Azure notebooks.

## Create a New Notebook in Azure Notebooks

One of the ways to use the Azure Machine Learning SDK for Python is with Azure notebooks. In the introduction we saw how to get started with Azure notebooks. Here we will show how to create a new notebook to get started.

In Azure notebooks, when you select Run to start your project, it opens Jupyter Notebooks interface. From within the Jupyter Notebooks interface, create a new notebook with Python 3.6 kernel as shown.

![create a new notebook](./media/new_notebook.png)

Now you are ready to write your code in the notebook.

## Create AML Compute Cluster

Azure Machine Learning Compute is a service for provisioning and managing clusters of Azure virtual machines for running machine learning workloads. In the introduction we saw how to create Azure Machine Learning workspace in your subscription. The following steps create a new Aml Compute in the workspace, if it doesn't already exist. This compute target will be used to run all the pipelines.

```python
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

aml_compute_target = "aml-compute"

try:
    # Look to see if the compute target already available in the workspace (ws)
    aml_compute = AmlCompute(ws, aml_compute_target)
    print("found existing compute target.")
except ComputeTargetException:
    print("creating new compute target")
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_D2_V2",
                                                                min_nodes = 1, 
                                                                max_nodes = 1)    
    aml_compute = ComputeTarget.create(ws, aml_compute_target, provisioning_config)
    aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
```

## Create the Run Configuration

Run configuration defines enviroment needed to run the piplines on the above created compute target. In this example, we will be using the numpy, pandas and the scikit-learn libraries for data preparation, modeling training, and batch predictions tasks. These required libraries are included in the environment as application dependencies.

```python
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE

# Create a new runconfig object
run_amlcompute = RunConfiguration()

# Use the cpu_cluster you created above
run_amlcompute.target = aml_compute_target

# Enable Docker
run_amlcompute.environment.docker.enabled = True

# Set Docker base image to the default CPU-based image
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_amlcompute.environment.python.user_managed_dependencies = False

# Auto-prepare the Docker image when used for execution (if it is not already prepared)
run_amlcompute.auto_prepare_environment = True

# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
    'numpy',
    'pandas',
    'scikit-learn',
    'sklearn_pandas'
])
```

## Creating a Pipeline for Repeatable Data Prep and Model Training

In this section we will describe how to use the Azure notebooks to build repeatable data preparation and model training pipeline with explicit dependency between the two pipeline steps.

### Create Data Preparation Pipeline Step

In the data preparation pipeline step, we take the raw input data, process the input data, and output the processed data that will be used in the model training step.

![data prep pipeline step](./media/data_prep.png)

#### Create a DataReference for the raw input data

Assuming that you have uploaded the raw input data file('s) to the default datastore in your workspace. You can first get the default datastore associated with workspace, and then create a DataReference for the raw input data by specifying the path of the raw input data file('s) in the datastore. You will pass this DataReference as input to your Data Prep Pipeline step.

```python
from azureml.data.data_reference import DataReference

# Get reference to the default data store in your workspace
def_blob_store = ws.get_default_datastore()

# Create a DataReference to the raw data input file
raw_data = DataReference(datastore=def_blob_store, 
                                      data_reference_name="raw_data", 
                                      path_on_datastore=".../...")
```

#### Create a PipelineData object for the processed data

The intermediate data (or output of a Step) is represented by PipelineData object. PipelineData can be produced by one step and consumed in another step by providing the PipelineData object as an output of one step and the input of one or more steps. Thus, to save the processed data / output from the Data Prep Pipeline step, you need to create a PipelineData object. We will use the default datastore to save the processed data.

```python
from azureml.pipeline.core import PipelineData

# Create the PipelineData object to host the processed data
processed_data = PipelineData('processed_data', datastore=def_blob_store)
```

#### Create the Data Prep Pipeline Step object

In this example, we will create a PythonScriptStep object that will run the code in the specified python script file as part of the Data Prep Pipeline Step execution. Your script, `process.py`, can define custom input parameters, such as `process_mode` that is unique to your needs. In this example, since the Data Prep pipeline step will be used to process input data at both training and inference time, the script file expects a custom input parameter named `process_mode` to distinguish the context in which it is called. We will look at the code to create the pipeline step, followed by an example python script file that will be used in the step.

Here are some key parameters used to create the pipeline step:

- The `source_directory` is the path to the python file `process.py`
- The script takes three arguments as inputs: process_mode: 'train' or 'inference', input location, and output location
- This step is created to be used in the **Data Prep - Model Train** pipeline, thus process_mode is set to 'train'
- inputs specify the DataReference object where the raw input file(s) are available
- ouputs takes the PipelineData object to save the intermediate processed data generated as part of this step
- Specify the compute target and run configuration

```python
from azureml.pipeline.steps import PythonScriptStep

# Create the Data Prep Pipeline Step Object
dataPrepStep = PythonScriptStep(
    name="process_data_step",
    source_directory="...",
    script_name="process.py", 
    arguments=["--process_mode", 'train',
               "--input", raw_data,
               "--output", processed_data],
    inputs=[raw_data],
    outputs=[processed_data],
    compute_target=aml_compute,
    runconfig=run_amlcompute
)

```




## Creating a pipeline for repeatable data prep and batch scoring using Azure Notebooks
