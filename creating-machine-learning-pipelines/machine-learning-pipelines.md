# Overview of machine learning pipelines using the Azure Machine Learning SDK

## What are machine learning pipelines?

Machine learning pipelines are cyclical and iterative in nature that facilitate both continuous improvement of model performance and deploying and making inferences on the best performing model to date. The pipelines comprise of distinct steps, for example, data preparation, model training, and batch predictions. Often data scientists, data engineers and IT professionals need to collaborate on building robust, scalable, and reusable machine learning pipelines.

The following diagram shows an example pipeline:

![azure machine learning piplines](./media/pipelines.png)

The [Azure Machine Learning SDK for Python](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/?view=azure-ml-py) allows you to create ML pipelines, and also submit and track individual pipeline runs. You can build reusable pipelines that optimize your specific workflows and allows you to focus on your expertise, for example machine learning, instead of the infrastructure to build and manage the pipelines.

The purpose of this article to show how to build an example machine learning pipeline work flow, that includes repeatable data preparation, model training and batch predictions using the Azure Machine Learning SDK for Python within Azure notebooks.

## Creating a pipeline for repeatable data prep and model training using Azure Notebooks

## Creating a pipeline for repeatable data prep and batch scoring using Azure Notebooks
