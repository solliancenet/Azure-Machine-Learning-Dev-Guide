# Getting your environment set up

Whether you are using the Azure Machine Learning service SDK on your local machine or a remote virtual machine, there are some configuration steps you must complete to start using your workspace. This article guides you through how to create your Azure Machine Learning service workspace, using a variety of options, and how to set up your development environment so you can start using the [available tools](./tools.md) to build, train, and deploy your models.

## Azure Machine Learning service workspace

To use Azure Machine Learning service, you need an [Azure Machine Learning service workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-workspace). This workspace is the top-level resource for the service and provides you with a centralized place to work with all the artifacts you create.

When you create a workspace the following Azure resources are added automatically (if they're regionally available):

- [Azure Container Registry](https://azure.microsoft.com/services/container-registry/): To minimize costs, ACR is **lazy-loaded** until deployment images are created.
- [Azure Storage](https://azure.microsoft.com/services/storage/)
- [Azure Application Insights](https://azure.microsoft.com/services/application-insights/)
- [Azure Key Vault](https://azure.microsoft.com/services/key-vault/)

### Option 1: Azure portal

Follow these steps to create your workspace through the Azure portal.

1. Sign in to the [Azure portal](https://portal.azure.com/). If you do not have an Azure subscription, you can get started with a [free account](https://aka.ms/AMLFree) today.
2. In the upper-left corner of the portal, select **+ Create a resource**.

   ![The Create a resource menu option is highlighted in the Azure portal.](media/azure-create-resource.png 'Create a resource')

3. In the search bar, enter `machine learning service workspace`. Select the **Machine Learning service workspace** search result.

   ![The term 'machine learning service workspace' contains search results. The Machine Learning service workspace result is highlighted.](media/azure-mlservice-workspace-search.png 'Machine learning service workspace search results')

4. In the **Machine Learning service workspace** pane, scroll to the bottom and select **Create** to begin.
5. In the **ML service workspace** pane, configure your workspace.

   | Field          | Description                                                                                                                                                                                                                                |
   | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | Workspace name | Enter a unique name that identifies your workspace. In this example, we use **aml-guide-workspace**. Names must be unique across the resource group. Use a name that's easy to recall and differentiate from workspaces created by others. |
   | Subscription   | Select the Azure subscription that you want to use.                                                                                                                                                                                        |
   | Resource group | Use an existing resource group in your subscription, or enter a name to create a new resource group. A resource group is a container that holds related resources for an Azure solution. In this example, we use **aml-guide**.            |
   | Location       | Select the location closest to your users and the data resources. This location is where the workspace is created.                                                                                                                         |

   ![The Machine Learning Service Workspace creation dialog is displayed with the previously described fields.](media/azure-ml-service-workspace-create-dialog.png 'Create Machine Learning Service Workspace dialog')

6. Select **Review + Create**. Verify everything looks correct, then select **Create** to begin the creation process. It can take a few moments to create the workspace.
7. You will be redirected to the Overview pane, which shows the deployment status. When your deployment is complete, select **Go to resource**.

   ![The Azure Machine Learning Service Workspace deployment status page is displayed, and the Go to resource button is highlighted.](media/azure-ml-service-workspace-deployment.png 'Machine Learning Service Workspace deployment status')

### Option 2: Machine Learning CLI

The Azure Machine Learning CLI is an extension to the [Azure CLI](https://docs.microsoft.com/cli/azure/?view=azure-cli-latest), a cross-platform command-line interface for the Azure platform. This extension provides commands for working with the Azure Machine Learning service. You can use it to automate tasks, such as creating an Azure Machine Learning service workspace.

You can perform these steps within Azure Cloud Shell, or from the [Azure CLI](https://docs.microsoft.com/cli/azure/?view=azure-cli-latest).

To use Azure Cloud Shell, select the **Cloud Shell** button at the top of the Azure portal. Ensure you have selected **Bash** for the Cloud Shell environment.

![Azure Cloud Shell link.](media/azure-cloud-shell.png 'Cloud Shell')

1. Install the Machine Learning CLI extension with the following command:

   ```bash
   az extension add -n azure-cli-ml
   ```

2. Verify that the extension has been installed:

   ```bash
   az ml -h
   ```

3. If you do not already have one, create an Azure resource group. Replace `myresourcegroup` with the name you want to assign to your resource group, and `westus2` with your desired resource group location:

   ```bash
   az group create -n myresourcegroup -l westus2
   ```

   > For a list of valid location names, execute: `az account list-locations -o table`

4. Create your Azure Machine Learning service workspace (see [az ml workspace create](https://docs.microsoft.com/en-us/cli/azure/ext/azure-cli-ml/ml/workspace?view=azure-cli-latest#ext-azure-cli-ml-az-ml-workspace-create)), replacing `myworkspace` with your desired globally unique workspace name, and `myresourcegroup` with the name of your resource group:

   ```bash
   az ml workspace create -w myworkspace -g myresourcegroup
   ```

After the workspace creation completes, the command will output your workspace details you can save for future reference.

![The Azure Cloud Shell is displayed with output from running the AML workspace creation command.](media/aml-workspace-script-cloud-shell.png 'CLI output')

### Option 3: Python SDK

Creating a workspace using the Python SDK is very useful for machine learning workspace creation and deployments from notebooks. For instance, you can create a new Azure Machine Learning service workspace from a local Jupyter notebook, an Azure Notebook, Azure Databricks notebook, HDInsight Spark notebook, etc., train your model, then deploy the model through the Python SDK.

#### Setup your Python environment and install the Python SDK

1. Create an isolated Python environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Anaconda](https://www.anaconda.com/), or [Python virtualenv](https://virtualenv.pypa.io/en/stable/). Be sure to select Python version 3.7 or greater.
2. Install the core components of the Machine Learning SDK with Jupyter notebook capabilities:

   ```shell
   pip install --upgrade azureml-sdk[notebooks]
   ```

3. Install the following packages so you can run the Azure machine Learning tutorials:

   ```shell
   conda install -y cython matplotlib pandas
   ```

> If you are using a Data Science Virtual Machine (DSVM), you do not need to install the Machine Learning SDK. Create an [Ubuntu DSVM](https://docs.microsoft.com/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro).

> If you are using Azure Databricks, [follow these instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#azure-databricks) to install the SDK and configure your cluster.

#### Create a workspace with the SDK

With your Python environment set up and the Python SDK installed, follow these steps to create a workspace with the SDK:

1. Create and/or cd to the directory you want to use for the quickstart and tutorials.

2. To launch Jupyter Notebook, enter this command:

   ```shell
   jupyter notebook
   ```

3. In the browser window, create a new notebook by using the default `Python 3` kernel.

4. To display the SDK version, enter and then execute the following Python code in a notebook cell:

   ```python
   import azureml.core
   print(azureml.core.VERSION)
   ```

5. Find a value for the `<azure-subscription-id>` parameter in the [subscriptions list in the Azure portal](https://ms.portal.azure.com/#blade/Microsoft_Azure_Billing/SubscriptionsBlade). Use any subscription in which your role is owner or contributor. For more information on roles, see [Manage access to an Azure Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-assign-roles) article.

   ```python
   from azureml.core import Workspace
   ws = Workspace.create(name='myworkspace',
                         subscription_id='<azure-subscription-id>',
                         resource_group='myresourcegroup',
                         create_resource_group=True,
                         location='eastus2'
                        )
   ```

   When you execute the code, you might be prompted to sign into your Azure account. After you sign in, the authentication token is cached locally.

6. To view the workspace details, such as associated storage, container registry, and key vault, enter the following code:

   ```python
   ws.get_details()
   ```

#### Write a configuration file

Save the details of your workspace in a configuration file to the current directory. This file is called _.azureml/config.json_.  
This workspace configuration file makes it easy to load the same workspace later. You can load it with other notebooks and scripts in the same directory or a subdirectory using the code `ws=Workspace.from_config()`.

```python
# Create the configuration file.
ws.write_config()

# Use this code to load the workspace from
# other scripts and notebooks in this directory.
# ws = Workspace.from_config()
```

This `write_config()` API call creates the configuration file in the current directory. The _config.json_ file contains the following:

```json
{
  "subscription_id": "<azure-subscription-id>",
  "resource_group": "myresourcegroup",
  "workspace_name": "myworkspace"
}
```

> [!TIP]
> To use your workspace in Python scripts or Jupyter Notebooks located in other directories, copy this file to that directory. The file can be in the same directory, a subdirectory named _.azureml_, or in a parent directory.

Alternatively, you may download the configuration file (config.json) that will be used by your notebooks to interact with your Machine Learning service workspace. The file should be saved to the top-level parent folder that contains your Jupyter notebooks. To download a generated configuration file, go to the Overview blade of your Azure Machine Learning service workspace in the Azure portal and select the **Download config.json** file. Or, you can [create your own configuration file](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace#write-a-configuration-file) manually.

### Azure Notebooks

Text

### Visual Studio Code

Text

## Next steps

- [Reference link]()
- [Reference link]()

Read next: [Related article]()
