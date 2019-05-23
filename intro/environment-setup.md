# Getting your environment set up

Whether you are using the Azure Machine Learning service SDK on your local machine or a remote virtual machine, there are some configuration steps you must complete to start using your workspace. This article guides you through how to create your Azure Machine Learning service workspace, using a variety of options, and how to set up your development environment so you can start using the [available tools](./tools.md) to build, train, and deploy your models.

## Create your Azure Machine Learning service workspace

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

Azure Notebooks gives you a fully managed environment in which you can develop and run Jupyter notebooks and easily share your projects with others. You do not need to sign in to start using Azure Notebooks, but any changes you make to notebooks or data files will not persist. This works well for scenarios where someone just needs to run a notebook, like as part of a tutorial or demonstration, without needing to sign in. However, if you want to Azure Notebooks to retain your projects across sessions, you must sign in with either a Microsoft account or a "Work or School" account. When the account used for Azure Notebooks is also associated with an Azure Subscription, you gain additional benefits such as running notebooks on more powerful servers, creating private notebooks, and granting permissions to notebooks to individual users.

[Learn more about signing in to Azure Notebooks](https://docs.microsoft.com/en-us/azure/notebooks/azure-notebooks-user-account) and which accounts you can use.

After signing into Azure Notebooks with your account for the first time, your account is automatically assigned a temporary user ID that begins with "anon-". As long as you have a user ID that begins with "anon-", Azure Notebooks prompts you to change it whenever you sign in. Setting your user ID is an important step because it is used as part of the URLs you use for others to view your profile and to share projects and notebooks. Here are the URL patterns. Notice how your user ID is used in the base path of each:

- `https://notebooks.azure.com/<user_id>`: Your profile page.
- `https://notebooks.azure.com/<user_id>/projects`: Your projects. You see all projects; other users see only your public projects.
- `https://notebooks.azure.com/<user_id>/projects/<project_id>`: Project files.
- `https://notebooks.azure.com/<user_id>/projects/<project_id>/clones`: Clones of a specific projects.
- `https://notebooks.azure.com/<user_id>/projects/<project_id>/html/<notebook>.ipynb`: The HTML preview of a specific notebook or file.

#### Profile and settings

In Azure Notebooks, your profile is how others can view public information about you, such as your display name, user ID, public email account, and any social profiles you wish to share. It also lists your recently used projects and any starred projects. Your profile is accessible by visiting `https://notebooks.azure.com/<user_id>`. You can also get to your profile page by selecting your user ID at the top-right corner of the Azure Notebooks site, then selecting **My Account**.

![The Azure Notebooks profile link located in the upper-right corner of the page is highlighted, as is the My Account link in the context menu underneath.](media/azure-notebooks-profile-link.png 'Azure Notebooks profile link')

Your profile page is publicly viewable and you can edit your profile to update what others see. You can also adjust Azure Notebooks site settings when you edit your profile. Just select the **Edit Profile Information** button on the top of your profile page.

![The Edit Profile Information button is highlighted on the profile page.](media/azure-notebooks-profile.png 'Azure Notebooks profile')

| Section              | Contents                                                                                                                                                                                                                                                                                                                                                                                                 |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Profile photo        | An image that's shown on your profile page.                                                                                                                                                                                                                                                                                                                                                              |
| Account Information  | Your display name, user ID, and public email account. The email account here provides other users a mean to contact you and can be different from the [account](azure-notebooks-user-account.md) you use to sign into Azure Notebooks itself.                                                                                                                                                            |
| Profile Information  | Your location, company, job title, web site, and a short description of yourself.                                                                                                                                                                                                                                                                                                                        |
| Social Profiles      | Your GItHub, Twitter, and Facebook IDs, if you wish to share them.                                                                                                                                                                                                                                                                                                                                       |
| Privacy Settings     | Provides two commands:<ul><li>**Export My Profile**: creates and downloads a _.zip_ file containing all the information that Azure Notebooks saves in your profile, including your photograph, profile information, and security logs.</li><li>**Delete My Account**: Permanently deletes all your personal information stored in Azure Notebooks.</li></ul>                                             |
| Enable Site Features | Allows you to control aspects of the behavior of Azure Notebooks:<ul><li>**Unified Frontend for Notebooks**: enables faster notebook startup and better persistence.</li><li>**Run in JupyterLab by default**: By default, Azure Notebooks provides a simple user interface that's suitable for most users. JupyterLab provides a richer but more complicated interface for experienced users.</li></ul> |

#### Project configuration

A project in Azure Notebooks is a collection of files, such as notebooks, data files, documentation, images, and other descriptive metadata. The project also defines the configuration of the underlying Linux virtual machine on which Jupyter notebooks run. The environment can be configured with specific setup commands. By defining the environment with the project, anyone who clones the project into their own Azure Notebooks account has all the information they need to recreate the necessary environment.

There are three ways to configure the environment of the underlying virtual machine in which your notebooks run:

- Include a one-time initialization script
- Use the project's environment settings to specify setup steps
- Access the virtual machine through a terminal.

All forms of project configuration are applied whenever the virtual machine is started, and thus affects all notebooks within the project.

1. To start, select **Project Settings** at the top of the project page.

   ![The Project Settings button is highlighted.](media/azure-notebooks-project-settings-button.png 'Project Settings')

2. In the **Project Settings** dialog that appears, select the **Environment** tab, then under **Environment Setup Steps**, select **+ Add**.
3. The **+ Add** command creates a step that's defined by an operation, a target file you select in your project, and sometimes an additional option, like Python version, depending on which step you select.

   ![The Environment tab is selected, and the operation dropdown is highlighted and displayed.](media/azure-notebooks-add-environment-setup-step.png 'Add Environment Setup Step')

   | Operation        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
   | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | Requirements.txt | Python projects define their dependencies in a requirements.txt file. With this option, select the appropriate file from the project's file list, and also select the Python version in the additional drop-down that appears. If necessary, select **Cancel** to return to the project, upload or create the file, then return to the **Project Settings** > **Environment** tab and create a new step. With this step in place, running a notebook in the project automatically runs `pip install -r <file>` |
   | Shell script     | Use to indicate a bash shell script (typically a file with the _.sh_ extension) that contains any commands you wish to run to initialize the environment.                                                                                                                                                                                                                                                                                                                                                      |
   | Environment.yml  | A Python project that uses conda for managing an environment uses an _environments.yml_ file to describe dependencies. With this option, select the appropriate file from the project's file list.                                                                                                                                                                                                                                                                                                             |

4. When you are done adding steps, select **Save**.

You can also upload a one-time initialization script. The first time Azure Notebooks creates a server for the project, it looks for a file in the project called `aznbsetup.sh`. If this file is present, Azure Notebooks runs it. The output of the script is stored in your project folder as _.aznbsetup.log_.

#### Project terminal

When you are viewing a project, select **Terminal** to open a Linux terminal that gives you direct access to the server. You can run almost any terminal command you normally would if you SSH into a VM, such as inspecting processes, downloading data, managing files, and editing files using tools like vi and nano.

When you run `ls` in the home folder, you can see which environments exist on the virtual machine, such as _anaconda2_501_, _anaconda3_420_, _anaconda3_501_, _IfSharp_, and _R_. You can modify a specific environment by changing to its environment folder before running commands.

> Changes made to the server apply **only** to the **current session**, except for files and folders you create in the project folder itself. For example, editing a file in the project folder is persisted between sessions, but packages with `pip install` are not. To persist environment settings, follow the instructions in the previous section to **add an Environment Setup Step**.

If you do make environment changes through the terminal, you would likely do so to experiment with settings and package installs on the VM that you will add as one of your project's Environment setup steps (requirements.txt, shell script, or environment.yml file), as detailed in the previous section.

If you use `python` or `python3`, you invoke the system-installed versions of Python, which are not used for notebooks. You don't have permissions for operations like `pip install` either, so be sure to use the version-specific aliases, as highlighted in the screenshot below.

![The Terminal window is displayed with a built-in alias used for a pip install.](media/azure-notebooks-terminal.png 'Terminal')

These are the built-in aliases for the environments:

```bash
# Anaconda 2 5.3.0/Python 2.7: python27
python27 -m pip install <package>

# Anaconda 3 4.2.0/Python 3.5: python35
python35 -m pip install <package>

# Anaconda 3 5.3.0/Python 3.6: python36
python36 -m pip install <package>
```

### Visual Studio Code

Text

## Next steps

- [Reference link]()
- [Reference link]()

Read next: [Related article]()
