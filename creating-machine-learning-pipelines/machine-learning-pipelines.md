# Overview of machine learning pipelines using the Azure Machine Learning SDK

## What are machine learning pipelines?

Machine learning pipelines are cyclical and iterative in nature that facilitate both continuous improvement of model performance and deploying and making inferences on the best performing model to date. The pipelines comprise of distinct steps, for example, data preparation, model training, and batch predictions. Often data scientists, data engineers and IT professionals need to collaborate on building robust, scalable, and reusable machine learning pipelines.

The following diagram shows an example pipeline:

![azure machine learning piplines](./media/pipelines.png)

The [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/?view=azure-ml-py) allows you to create ML pipelines, and also submit and track individual pipeline runs. You can build reusable pipelines that optimize your specific workflows and allows you to focus on your expertise, for example machine learning, instead of the infrastructure to build and manage the pipelines.

The purpose of this article to show how to build an example machine learning pipeline work flow, that includes repeatable data preparation, model training and batch predictions using the Azure Machine Learning SDK for Python within Azure notebooks.

## Creating a pipeline for repeatable data prep and model training using Azure Notebooks

One of the ways to use the Azure Machine Learning SDK for Python is with Azure notebooks. In the introduction we saw how to get started with Azure notebooks, and how to create Azure Machine Learning workspace in your subscription. In this section we will describe how to use the Azure notebooks to build repeatable data preparation and model training pipeline with explicit dependency between the two pipeline steps.

### Create a new notebook

In Azure notebooks, when you select Run to start your project, it opens Jupyter Notebooks interface. From within the Jupyter Notebooks interface, create a new notebook with Python 3.6 kernel as shown.

![create a new notebook](./media/new_notebook.png)

Now you are ready to write your code in the notebook.

### Create Data Preparation Pipeline Step

In the data preparation pipeline step, we take the raw input data, process the input data, and output the processed data that will be used in the model training step.

![data prep pipeline step](./media/data_prep.png)

#### Create a DataReference for the raw input data

Assuming that you have uploaded the raw input data file('s) to the default datastore in your workspace. You can first get the default datastore associated with workspace, and then create a DataReference for the raw input data by specifying the path of the raw input data file('s) in the datastore. You will pass this DataReference as input to your Data Prep Pipeline step.

```python
from azureml.core import Workspace
from azureml.data.data_reference import DataReference

# Create your workspace instance from config.
ws = Workspace.from_config()

# Get reference to the default data store in your workspace.
def_blob_store = ws.get_default_datastore()

# Create a DataReference to the raw data input file
raw_data = DataReference(datastore=def_blob_store, 
                                      data_reference_name="raw_data", 
                                      path_on_datastore=".../...")
```

#### Create a PipelineData Object

The intermediate data (or output of a Step) is represented by PipelineData object. PipelineData can be produced by one step and consumed in another step by providing the PipelineData object as an output of one step and the input of one or more steps. Thus, to save the processed data / output from the Data Prep Pipeline step, you need to create a PipelineData object. We will use the default datastore to save the processed data.

```python
from azureml.pipeline.core import PipelineData

# Create the PipelineData object to host the processed data.
processed_data = PipelineData('processed_data', datastore=def_blob_store)

```

#### Create the Data Prep Pipeline Step Object

In this example, we will create a PythonScriptStep object that will run the code in the specified python script file as part of the Data Prep Pipeline Step execution. 

```python
from azureml.pipeline.steps import PythonScriptStep

# Create the Data Prep Pipeline Step Object.
dataPrepStep = PythonScriptStep(
    name="process_data_step",
    script_name="process.py", 
    arguments=["--process_mode", 'train',
               "--input", raw_data,
               "--output", processed_data],
    inputs=[raw_data],
    outputs=[processed_data],
    compute_target=aml_compute,
    runconfig=run_amlcompute,
    source_directory="..."
)

```


## Creating a pipeline for repeatable data prep and batch scoring using Azure Notebooks
