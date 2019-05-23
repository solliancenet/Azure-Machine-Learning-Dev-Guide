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

Visual Studio Code is an excellent choice when you want to use a lightweight integrated development environment (IDE) for Python development. The available extensions you can install adds additional capabilities, like code completion (IntelliSense), code snippets, spell checking, linting, and many more.

#### Install the Azure Machine Learning extension

The [Azure Machine Learning extension](https://aka.ms/vscodetoolsforai) for Visual Studio Code adds features that help you manage your Machine Learning service workspace, and train and deploy machine learning and deep learning models.

To install the extension, perform the following steps within Visual Studio Code:

1. In Visual Studio Code, select the **Extensions** menu item in the left-hand menu.
2. Type "azure machine learning" in the search box, then select **Azure Machine Learning** from the search results.

   ![The Extensions menu item is highlighted, and the search box contains azure machine learning.](media/vs-code-aml-extension.png 'Visual Studio Code extension search')

3. Click the **Install** button on either the search result or the extension information page.

When you install this extension, the [Azure Account extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azure-account) and the [Microsoft Python extension ](https://marketplace.visualstudio.com/items?itemName=ms-python.python) are also installed.

#### Set up your Python environment

You will need to [install Python version **3.7.3** or higher](https://www.python.org/downloads/) before you begin. If Visual Studio Code is open during the install, you will need to restart it after installing Python.

Before you can begin using the new Python capabilities in VS Code, you must select your Python interpreter:

1. Open the **Command Palette** in VS Code (`Ctrl+Shift+P`).
2. Start typing **Python: Select Interpreter** and select that command.

   ![The VS Code Command Palette is displayed with the Python Select Interpreter command selected in the search results.](media/vs-code-python-select-interpreter.png 'Python: Select Interpreter')

   > If you see an error stating that `python.pythonPath` cannot be set or does not exist, you may not have installed an interpreter, such as [Anaconda](https://www.anaconda.com/download/). Install Anaconda and restart Visual Studio Code before attempting to select the interpreter.

3. Select the interpreter you wish to use. If you installed Python 3.7.3 or higher, be sure to select that newer version. In this case, we selected the `Python 3.7.3 32-bit` interpreter.

   ![Select your desired Python interpreter from the list.](media/vs-code-python-interpreter-selected.png 'Python interpreters')

4. Start a new Extension search by typing "intellicode". Select the **Visual Studio IntelliCode** extension and install it. This will provide you with AI-assisted capabilities for IntelliSense in Python.

   ![The Visual Studio IntelliCode extension is selected.](media/vs-code-intellicode-extension.png 'Visual Studio IntelliCode extension')

Follow the instructions [here](https://code.visualstudio.com/docs/python/python-tutorial) for configuring VS Code for Python development.

For more information about editing, running, and debugging Python code, see the [Python hello-world tutorial](https://code.visualstudio.com/docs/python/python-tutorial).

### Visual interface

Because Azure Machine Learning services visual interface is a fully managed, cloud-hosted environment, there are not a lot of options for configuring the environment. However, you can define the compute targets for running your experiments as well as for deploying your trained models into web services.

Azure Machine Learning Compute is a managed-compute infrastructure that allows the user to easily create a single or multi-node compute. The compute is created within your workspace region as a resource that can be shared with other users in your workspace. The compute scales up automatically when a job is submitted, and can be put in an Azure Virtual Network. The compute executes in a containerized environment and packages your model dependencies in a [Docker container](https://www.docker.com/why-docker).

You can use Azure Machine Learning Compute to distribute the training process across a cluster of CPU or GPU compute nodes in the cloud. For more information on the VM sizes that include GPUs, see [GPU-optimized virtual machine sizes](https://docs.microsoft.com/azure/virtual-machines/linux/sizes-gpu).

Azure Machine Learning Compute has default limits, such as the number of cores that can be allocated. For more information, see [Manage and request quotas for Azure resources](https://docs.microsoft.com/azure/machine-learning/service/how-to-manage-quotas).

#### Create a new compute target for experiments

There are four ways you can create new compute targets for experiments: within visual interface, in the Azure portal, with the CLI, and using the Azure Machine Learning SDK.

##### Option 1: Visual interface

When you are ready to run an experiment in within the visual interface, you have the choice to select an existing compute target, or create a new one.

1. Select **Run** at the bottom of the experiment canvas.

   ![The Run button is highlighted on the bottom of the experiment canvas.](media/visual-interface-run.png 'Run')

2. When the **Setup Compute Targets** dialog appears, select **Create new**. This will give you the option of selecting the pre-defined compute configuration and entering a **New Compute Name**.

   ![The Setup Compute Target to Run Experiment dialog is displayed with Create new selected.](media/visual-interface-compute-target.png 'Setup compute target to run experiment')

3. Select **Run**. This will create the compute target and run your experiment.

##### Option 2: Azure portal

1. Sign in to the [Azure portal](https://portal.azure.com).
2. Open your Machine Learning service workspace, then select **Compute** on the left-hand menu.
3. Select **+ Add Compute**.

   ![The Compute left-hand menu item and Add Compute button are both highlighted.](media/aml-workspace-add-compute.png 'Compute')

4. Select **Machine Learning Compute** as the **Compute type**. Provide values for the required properties, especially **VM Family**, and the **maximum nodes** to use to spin up the compute.

   ![The Add Compute form is displayed with the previously described fields completed.](media/aml-workspace-add-compute-form.png 'Add Compute')

   > The visual interface can only run experiments on Machine Learning Compute targets. Other compute targets will not be shown. If you want to use other compute targets, you must use the SDK.

5. Select **Create**.

##### Option 3: CLI

You can use the Machine Learning CLI to create a persistent compute target for running visual interface experiments. The [instructions for using the CLI](#option-2-machine-learning-cli) is at the top of this page under the section for creating your AML workspace.

When you use the Machine Learning CLI, specify the **AMLcompute** target.

```bash
az ml computetarget create amlcompute -n cpu --min-nodes 1 --max-nodes 1 -s STANDARD_D3_V2
```

For more information, see [az ml computetarget create amlcompute](https://docs.microsoft.com/cli/azure/ext/azure-cli-ml/ml/computetarget/create?view=azure-cli-latest#ext-azure-cli-ml-az-ml-computetarget-create-amlcompute).

##### Option 4: Python SDK

When you create a compute target using the Python SDK, you can create an Azure Machine Learning compute environment on demand when you schedule a run, or as a persistent resource. When you are creating a compute environment for visual interface, you will need to create a persistent resource.

1. **Create and attach**: To create a persistent Azure Machine Learning Compute resource in Python, specify the **vm_size** and **max_nodes** properties. Azure Machine Learning then uses smart defaults for the other properties. The compute autoscales down to zero nodes when it isn't used. Dedicated VMs are created to run your jobs as needed.

   - **vm_size**: The VM family of the nodes created by Azure Machine Learning Compute.
   - **max_nodes**: The max number of nodes to autoscale up to when you run a job on Azure Machine Learning Compute.

   ```python
   from azureml.core.compute import ComputeTarget, AmlCompute
   from azureml.core.compute_target import ComputeTargetException

   # Choose a name for your CPU cluster
   cpu_cluster_name = "cpucluster"

   # Verify that cluster does not exist already
   try:
       cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
       print('Found existing cluster, use it.')
   except ComputeTargetException:
       compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                             max_nodes=4)
       cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

   cpu_cluster.wait_for_completion(show_output=True)
   ```

   You can also configure several advanced properties when you create Azure Machine Learning Compute. The properties allow you to create a persistent cluster of fixed size, or within an existing Azure Virtual Network in your subscription. See the [AmlCompute class](https://docs.microsoft.com/python/api/azureml-core/azureml.core.compute.amlcompute.amlcompute?view=azure-ml-py) for details.

   Or you can create and attach a persistent Azure Machine Learning Compute resource [in the Azure portal](#portal-create).

2. **Configure**: Create a run configuration for the persistent compute target.

   ```python
   from azureml.core.runconfig import RunConfiguration
   from azureml.core.conda_dependencies import CondaDependencies
   from azureml.core.runconfig import DEFAULT_CPU_IMAGE

   # Create a new runconfig object
   run_amlcompute = RunConfiguration()

   # Use the cpu_cluster you created above.
   run_amlcompute.target = cpu_cluster

   # Enable Docker
   run_amlcompute.environment.docker.enabled = True

   # Set Docker base image to the default CPU-based image
   run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

   # Use conda_dependencies.yml to create a conda environment in the Docker image for execution
   run_amlcompute.environment.python.user_managed_dependencies = False

   # Auto-prepare the Docker image when used for execution (if it is not already prepared)
   run_amlcompute.auto_prepare_environment = True

   # Specify CondaDependencies obj, add necessary packages
   run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])
   ```

#### Create a new compute target for web services

When you deploy a web service in visual interface, you must select a compute target that will host the web service. We recommend that you add a Kubernetes Service compute target to your workspace and [follow the instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-secure-web-service) to use SSL to secure the web service.

You may use the methods above to create a new Kubernetes Service compute target by specifying the compute type. Below is a screenshot of the Add Compute form for a Kubernetes Service compute target in the Azure portal:

![The Compute menu item is selected in the AML workspace, and Kubernetes Service is selected as the compute type.](media/visual-interface-create-kubernetes-compute-target.png 'Add Compute')

## Next steps

Please see the following additional references for configuring your development environment:

- [Use the CLI extension for Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/reference-azure-machine-learning-cli#resource-management)
- [What is the Azure Machine Learning SDK for Python?](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py)
- [Configure dev environments](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment)

Read next: [What is Azure Machine Learning?](./what-is-azure-machine-learning.md)
