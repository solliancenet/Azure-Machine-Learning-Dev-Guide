# Overview of Azure Machine Learning service architecture and concepts

Azure Machine Learning service provides a comprehensive environment you can use with a suite of [powerful tools](./tools.md) to perform data science and engineering processes. The components of this environment give you a centralized place to work with all the artifacts involved in the process.

## Components and concepts

The top-level component of the Azure Machine Learning (AML) service is the [workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-workspace). Your first step to using AML is to create a workspace within an Azure region and resource group (a logical grouping of Azure services). When you do this, the script creates all services alongside your workspace. Once created, you reference workspace any time you need to train, evaluate, monitor, or deploy a model.

Here is an example of creating a new workspace, using the Machine Learning CLI (command-line interface). See [Getting your environment set up](./environment-setup.md) for more information.

```bash
az ml workspace create -w myworkspacename -g myresourcegroupname
```

The following diagram shows the high-level taxonomy of the workspace:

![The high-level taxonomy of the Azure Machine Learning workspace](./media/azure-machine-learning-taxonomy.png)

The diagram includes the following components:

- A workspace may contain [Notebook VMs](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-run-cloud-notebook), which are cloud resources pre-configured with a Python environment necessary to run Azure Machine Learning.
- [User roles](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-assign-roles) enable you to share your workspace with other users, teams, or projects.
- [Compute targets](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-compute-target) are compute resources where you run your experiments or host your service deployment. The target may be your local machine or a cloud-based resource.
- When you create the workspace, [associated resources](#associated-workspace-resources) are also created for you.
- [Experiments](../modeling/feature-engineering-training-evaluation-selection/model-training/aml-experiment-runs.md) are training runs you use to build your models. You can create and run experiments with
  - The [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py).
  - The [automated machine learning experiments](../modeling/simplify-process-with-automated-ml/README.md) section in the Azure portal.
  - The [visual interface](./tools.md#visual-interface) for drag-and-drop model building and deployment.
- [Pipelines](../creating-machine-learning-pipelines/machine-learning-pipelines.md) are reusable workflows for training and retraining your model.
- [Datasets](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-explore-prepare-data) aid in the management of the data you use for model training and pipeline creation.
- Once you have a model you want to deploy, you create a registered model.
- Use the registered model and a scoring script to create a [deployment](../model-deployment/deployment-target-options.md).

## Associated workspace resources

Beyond the workspace, there are a few related Azure resources that are used to manage, monitor, and secure the workspace components. These are created for you automatically when you create a new workspace. However, you can choose to use existing Azure services in place of creating new versions.

- [Azure Container Registry](https://azure.microsoft.com/services/container-registry/): Registers docker containers that you use during training and when you deploy a model. To minimize costs, ACR is **lazy-loaded** until deployment images are created.
- [Azure Storage account](https://azure.microsoft.com/services/storage/): Is used as the default data store for the workspace. Jupyter notebooks that are used with your notebook VMs are stored here as well.
- [Azure Application Insights](https://azure.microsoft.com/services/application-insights/): Allows you to monitor your models and view metrics on their usage.
- [Azure Key Vault](https://azure.microsoft.com/services/key-vault/): Securely stores secrets and other sensitive information used by your workspace and any compute targets.

## High-level model workflow

![The stages shown are Train, Package, Validate, Deploy, and Monitor. An arrow labeled Retrain goes back to Train from Monitor.](media/aml-model-workflow.png 'Azure Machine Learning model workflow')

### 1 - Train

At the core of the modern data science process is [training, evaluating, and selecting machine learning models](../modeling/feature-engineering-training-evaluation-selection/README.md). After selecting an algorithm for your model, you train it with data that has been evaluated and prepared with the transformations and features required for training. At a very high level, training a model with Azure Machine Learning service involves the following steps:

- Using your favorite [Python environment](./environment-setup.md), create a machine learning training script along with any associated files. Specify the directory that contains these files, as well as an experiment name. Alternately, use visual interface for a code-free experience.
- Create and configure a compute target for executing the training. During training, the complete directory copies to the compute target before the training script executes.
- Submit the training scripts to the configured compute target. Afterward, the script starts running within the environment and has access to read from and write to [datastores](../data-acquisition-understanding/accessing-data.md#working-with-datastores). Each execution saves a record of the run within the workspace, grouped under experiments.

Use the automated machine learning feature to select the best model during training automatically. This feature automates experimenting with multiple combinations of parameter values, also referred to as hyperparameter tuning, to accelerate the model training process and keeps a record of the outcomes to help identify potential areas of improvement more quickly.

### 2 - Package

After you have trained your model and have identified the best version, you package it along with all the components you need to use the model, into an image. The image can be either a Docker image or an FPGA image used to deploy your model to a field-programmable gate array. The image saves to the image registry in your workspace. The registry provides a centralized place to store your models so they can be easily copied to new deployment targets, as well as versioned.

### 3 - Validate

Model validation is used to calculate the accuracy of a model. Validation happens during the training process to make sure your chosen algorithm is performing as expected. It is also conducted periodically to ensure your model is still performing well over time with new data. Azure Machine Learning service allows you to query your experiments for logged metrics from current and past runs. Use the metrics to determine whether the run had the desired outcome. If not, begin the retraining process by starting over at step one.

### 4 - Deploy

When you want your model to be available for on-demand access over the web or on an IoT device, you use the Azure Machine Learning SDK to deploy it as a web service in Azure, an FPGA, or to an IoT Edge device. The image that you created when you packaged the model is used to copy instances of the model, scoring script, and any dependencies to your deployment target. Your options for deploying the model as a web service are Azure Container Instance (ACI) and Azure Kubernetes Service (AKS).

### 5 - Monitor

Monitor for changes in the distribution of data between the training dataset and inference data of your deployed model. These changes are sometimes called [data drift](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-data-drift) and indicate degraded prediction performance over time due to how the input data changes during this period. When you detect degraded model performance, the next step is to retrain your model with new data, thus creating a new version of the model. If your model is deployed to a web service or IoT devices, then you would use the new version of the model to redeploy it to those endpoints. The Azure Machine Learning SDK provides tools you can use to redeploy with minimal interruption of these services and endpoints.

## Next steps

- [Quickstart guide for running a cloud notebook on a Notebook VM](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-run-cloud-notebook)
- [How to assign user roles](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-assign-roles)
- [Compute targets](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-compute-target)
- [Running model experiments](../modeling/feature-engineering-training-evaluation-selection/model-training/aml-experiment-runs.md)
- [Azure Machine Learning SDK for Python](https://docs.microsoft.com/python/api/overview/azure/ml/intro?view=azure-ml-py)
- [Automated machine learning experiments](../modeling/simplify-process-with-automated-ml/README.md)
- [Visual interface](./tools.md#visual-interface)
- [Building reusable workflows with Pipelines](../creating-machine-learning-pipelines/machine-learning-pipelines.md)
- [Model deployment target options](../model-deployment/deployment-target-options.md)

Read next: [What tools are used to do data engineering, data science, and AI?](./tools.md)
