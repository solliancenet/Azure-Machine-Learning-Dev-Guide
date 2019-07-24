# What tools are used to do data engineering, data science, and AI?

## Introducing the notebook paradigm

Executable, or interactive, notebooks have a long history in science and academia. Notebooks were traditionally provided by applications such as [MATLAB](https://www.mathworks.com/products/matlab.html) and [Wolfram Mathematica](https://www.wolfram.com/mathematica/) to help scientists, students, professors, and mathematicians create self-documenting notebooks that others can use to reproduce experiments. To accomplish this, notebooks contain a combination of runnable code, output, formatted text, and visualizations. Over the past several years, web-based interactive notebooks have gained popularity with data scientists and data engineers to conduct exploratory data analysis and model training using several languages, such as Python, Scala, SQL, R, and others. The most popular notebooks in use today are [Jupyter](http://jupyter.org/), [Databricks Notebooks](https://docs.azuredatabricks.net/user-guide/notebooks/index.html), [R Markdown](jhttp://rmarkdown.rstudio.com/), and [Apache Zeppelin](https://zeppelin.apache.org/).

Notebooks are made up of one or more cells that allow for the execution of the code snippets or commands within those cells. They store commands and the results of running those commands. The image below shows a notebook that contains a cell on top that displays formatted text using markdown, followed by a cell containing Python code that gets executed to display an output. In this case, we chose a line chart visualization in place of the default text or table output. When the notebook is saved and shared with others, any outputs can be included so others can see the intended outcome of each cell.

![A screenshot of a notebook is displayed showing two cells. The first cell contains formatted text, and the second cell contains Python code and an output containing a line chart visualization.](media/notebook-cells.png 'Sample notebook cells')

Notebooks can be run locally, on a notebook server, such as [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/), or in a hosted environment, such as [notebook VMs](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#notebookvm), [Azure Notebooks](https://notebooks.azure.com/), or [Azure Databricks](https://docs.microsoft.com/en-us/azure/azure-databricks/what-is-azure-databricks). Depending on your notebook environment, you can connect to several data sources and select from a large collection of open-source libraries and SDKs (such as the Azure Machine Learning SDK) supported by the notebook's kernel (Python, etc.) to accelerate your development efforts. When you execute the cells in your notebook, you must first connect to an execution engine or a cluster. The execution context can be powered by local compute resources, such as your laptop or an on-premises cluster, or remote resources like a VM or cluster in Azure. Being able to select the execution context gives you the flexibility to change the environment in which the embedded code can execute, allowing you to share the notebook across different environments and scale-out computational workloads as needed. In most cases, you will run your notebooks in a cluster environment when performing model training. This allows you to execute more complex computational processing and use larger data sets than you would be able to from your own machine.

Notebooks provide many advantages over traditional IDEs. They offer an environment that allows for exploration, documentation, collaboration, and visualization. When a data scientist creates and shares it with a colleague, they are sharing notes and insights about the data with access to all of the queries, formulas, visualizations, and models. This enables interactive conversations and further exploration, with simple reproducibility by anyone running the notebook in the same or similar environment, without others needing to know a sequence of shell commands and environment variables known only to the original author. This collaborative knowledge exchange within an easy to share self-contained package is far more valuable than merely sharing a static, final report. However, if you are used to developing software and applications using your favorite IDE, then you will realize that there are some disadvantages to using notebooks in place of a more traditional development platform. For example, you cannot set breakpoints and run in debug mode, allowing you to step through the code and inspect object and environment states during execution.

## Azure Notebooks and Jupyter notebooks

The one way to use the Azure Machine Learning service SDK is with Python-based Jupyter notebooks. In this section, we will describe how to get up and running using Jupyter notebooks and Azure ML within the free Azure Notebooks service, or with one of the Jupyter notebook server options.

### Using Jupyter notebook servers

If you choose to run Jupyter notebooks on your own notebook server, you can get started by either using a managed notebook server in the cloud or installing and running the notebook server locally.

#### Option 1: Use a cloud-based notebook server

With this option, you can quickly get started with Azure Machine Learning service by configuring a managed Jupyter notebook server in the cloud. The environment is provided by a [notebook VM](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#notebookvm), which is a secure, cloud-based Azure workstation that comes configured with a fully prepared machine learning environment and a Jupyter notebook server (JupyterLab). You can immediately start running notebooks and logging experiments in your ML workspace. There is no need to install any additional software or libraries in this environment. To get started, perform the following steps:

1. Sign in to the [Azure portal](https://portal.azure.com/) and open your Azure Machine Learning service workspace. [Create your workspace](./environment-setup.md#create-your-azure-machine-learning-service-workspace) if you have not already done so.
2. On your workspace page, select **Notebook VMs** on the left-hand menu.
3. Select **+ New** to create a notebook VM.

   ![The Notebook VMs link is highlighted on the left-hand menu of the Azure ML Service Workspace, and the New button is highlighted within the Notebook VMs blade.](media/new-notebook-vm-link.png 'New Notebook VM')

4. Provide a name for your VM, select a VM size, then select **Create**.

   ![The New Notebook VM form is displayed with the previously described fields.](media/new-notebook-vm.png 'New Notebook VM')

5. Wait until the status changes to **Running**. This will take approximately 4-5 minutes.
6. You will see the new VM in the list of Notebook VMs after it is created. Select the **Jupyter** link in the **URI** column for your new VM.

   ![The Jupyter link for the new Notebook VM is highlighted.](media/notebook-vm-jupyter-link.png 'Notebook VMs')

   Clicking this link starts your notebook server and opens the Jupyter notebook in a new web browser tab. Only the person who created the VM can use this link.

7. On the Jupyter notebook webpage, the top folder name is your username. Select this folder.
8. The samples folder name includes a version number, for example, **samples-1.0.33.1**. Select the samples folder.
9. Select the quickstart folder, then open the **01.run-experiment.ipynb** file.

   > The config.json file that is located in the top-level parent folder was created for you with details of your Azure Machine Learning service workspace. All notebooks in the samples folder use this file to connect to your workspace. If you want to download the file to use in other environments, you can go to the Overview blade of your Azure Machine Learning service workspace and select the **Download config.json** file. This config file is what the following command in the notebooks use to load your workspace configuration: `Workspace.from_config()`.

10. After running all the cells in the notebook, the last cell provides you with a link to view the experiment you just ran, within your AML workspace. From here you can go to your experiment and view this and all the other runs.

    ![The last cell contains a link to the Azure Portal.](media/aml-experiment-link.png 'Link on last cell')

    ![AML Workspace experiment run.](media/aml-experiment-run.png 'Experiment run')

    > The plots of logged values are automatically created in the workspace whenever you log multiple values within the same name parameter.

#### Option 2: Use your own notebook server

If you do not want to set up a Notebook VM and wish to run Jupyter notebooks on your own notebook server hosted on your local machine or any VM of your choosing, then you must perform the following steps:

1. Sign in to the [Azure portal](https://portal.azure.com/) and open your Azure Machine Learning service workspace. [Create your workspace](./environment-setup.md#create-your-azure-machine-learning-service-workspace) if you have not already done so.
2. Create an isolated Python environment using [Miniconda](https://docs.conda.io/en/latest/miniconda.html), [Anaconda](https://www.anaconda.com/), or [Python virtualenv](https://virtualenv.pypa.io/en/stable/). Be sure to select Python version 3.7 or greater.
3. Install the core components of the Machine Learning SDK with Jupyter notebook capabilities:

   ```shell
   pip install --upgrade azureml-sdk[notebooks]
   ```

4. Install the following packages so you can run the Azure machine Learning tutorials:

   ```shell
   conda install -y cython matplotlib pandas
   ```

5. Download or create a configuration file (config.json) that will be used by your notebooks to interact with your Machine Learning service workspace. The file should be saved to the top-level parent folder that contains your Jupyter notebooks. To download a generated configuration file, go to the Overview blade of your Azure Machine Learning service workspace in the Azure portal and select the **Download config.json** file. Alternately, you can [create your own configuration file](https://docs.microsoft.com/en-us/azure/machine-learning/service/setup-create-workspace#write-a-configuration-file) manually.

> You do not need to install the Machine Learning SDK if you are using a Data Science Virtual Machine (DSVM). Create an [Ubuntu DSVM](https://docs.microsoft.com/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro).

> If you are using Azure Databricks, [follow these instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#azure-databricks) to install the SDK and configure your cluster.

#### Use the workspace on your own notebook server

Find the directory in which your workspace configuration file is located, and start a notebook or create a new script. Add and execute the following code to use the basic SDK APIs to track experiment runs.

These steps show you how to:

1. Create an experiment in the workspace.
2. Log a single value into the experiment.
3. Log a list of values into the experiment.

```python
from azureml.core import Experiment

# Create a new experiment in your workspace.
exp = Experiment(workspace=ws, name='myexp')

# Start a run and start the logging service.
run = exp.start_logging()

# Log a single number.
run.log('my magic number', 42)

# Log a list (Fibonacci numbers).
run.log_list('my list', [1, 1, 2, 3, 5, 8, 13, 21, 34, 55])

# Finish the run.
run.complete()
```

Wait until the run completes, then execute the following to display a URL that shows the experiment run in the Azure portal:

```python
print(run.get_portal_url())
```

Copy and paste the URL from the command above into a new browser window. The URL will direct you to the Azure portal and open the experiment you just ran. Below is an example chart and attributes for the first run of the experiment.

![This screenshot shows the output of the above experiment as viewed from the Azure portal.](media/simple-experiment-run.png 'Logged values from experiment')

### Using the Azure Notebooks service

To start using notebooks for your own experimentation and model training, you can quickly get up and running by using Jupyter notebooks in the hosted Microsoft [Azure Notebooks](https://notebooks.azure.com/) environment. Azure Notebooks provides a globally available environment for running Jupyter notebooks within your own projects that you can share with others. A project in Azure Notebooks is essentially a configuration of the underlying Linux virtual machine in which Jupyter notebooks run, along with a file folder and descriptive metadata. You can choose to run your notebooks on a free compute tier, or on an Azure virtual machine, such as an Azure Data Science Virtual Machine (DSVM) using the [Data Science Virtual Machine for Linux (Ubuntu) image](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro).

Azure Notebooks is ready to work with [Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/), thanks to its pre-configured environment. There are two Azure ML-related projects you can use to get started in Azure Notebooks quickly. Below, we walk you through two options for cloning a project into your Azure Notebooks account so you can start experimenting with both projects.

#### Option 1: Clone an existing project with the Clone button

The first project is [Getting Started](https://notebooks.azure.com/azureml/projects/azureml-getting-started), provided by the Microsoft Azure Machine Learning Team ([azureml](https://notebooks.azure.com/azureml)). To clone this project, perform the following steps:

1. Sign into [Azure Notebooks](https://notebooks.azure.com/).
2. Navigate to the [Getting Started](https://notebooks.azure.com/azureml/projects/azureml-getting-started) project.
3. Select the **Clone** button at the top of the page.

   ![The Clone button is highlighted at the top of the page.](media/azure-notebooks-clone-button.png 'Clone Azure Notebooks project')

4. In the Clone Project dialog that appears, optionally change the project name and Project ID (the custom ID used to create a simple URL to your project for sharing), specify whether to make the project publicly visible to others, then select **Clone**.
5. After a few seconds, a copy of the project will be available for you to configure and run within your own Azure Notebooks account.

#### Option 2: Upload project from a GitHub repository

The second project you can clone is the [Azure Machine Learning service Tutorial](https://github.com/Azure/MachineLearningNotebooks/tree/master/tutorials) hosted on GitHub. This project contains additional tutorials you can run that covers various machine learning scenarios. To clone this project, perform the following steps:

1. Sign into [Azure Notebooks](https://notebooks.azure.com/).
2. Select **My Projects** to navigate to the projects dashboard.
3. Select the **Upload GitHub Repo** (the up arrow) button to open the Upload GitHub Repository dialog.
4. In the dialog, enter `Azure/MachineLearningNotebooks` in the **GitHub repository** field, clear **Clone recursively** since it is not needed for this project, provide a name for the project in the **Project Name** field, such as "Azure ML Services", provide an identifier in the **Project ID** field, clear **Public** if you want your clone to be private, then select **Import**.

   ![The Upload GitHub Repository dialog is displayed with the previously described fields filled out.](media/azure-notebooks-upload-github-repo.png 'Upload GitHub Repository')

5. After a couple minutes, Azure Notebooks automatically takes you to the new project's dashboard.

## Visual Studio Code with the Azure Machine Learning extension

[Visual Studio Code](https://code.visualstudio.com/docs/setup/setup-overview) is a powerful and flexible, yet lightweight source code editor that runs on your Windows, Linux, or macOS desktop. It comes with built-in support for JavaScript, TypeScript, and Node.js, but can be extended to support other languages and runtimes such as C++, C#, Java, Python, PHP, Go, .NET, and Unity.

Visual Studio Code has rapidly become one of the most popular free cross-platform integrated development environments (IDE) in use today. The primary reasons for this are its very lightweight size, which allows it to quickly run on almost any desktop hardware and its large extension marketplace that helps developers add capabilities for a wide range of development tasks.

The [Azure Machine Learning extension](https://aka.ms/vscodetoolsforai) for Visual Studio Code adds features that help you manage your Machine Learning service workspace, and train and deploy machine learning and deep learning models. When you install this extension, the [Azure Account extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azure-account) and the [Microsoft Python extension ](https://marketplace.visualstudio.com/items?itemName=ms-python.python) are also installed. These extensions allow you to connect to your Azure subscription and configure the Machine Learning service workspace and turns Visual Studio Code into a Python IDE. The Microsoft Python extension harnesses the power of Visual Studio Code to provide debugging, unit testing, linting, IntelliSense, and autocomplete capabilities to your Python scripts and applications. You can also easily switch between Python environments with the extension, including virtual and conda environments. See the [Python hello-world tutorial](https://code.visualstudio.com/docs/python/python-tutorial) for information about editing, running, and debugging Python code.

You need to [install Python version **3.7.3** or higher](https://www.python.org/downloads/) before you begin. If Visual Studio Code is open during the install, you need to restart it after installing Python.

Before you can begin using the new Python capabilities in VS Code, you must select your Python interpreter. You do this by opening the **Command Palette** in VS Code (`Ctrl+Shift+P`), typing **Python: Select Interpreter** and selecting that command. If you see an error stating that `python.pythonPath` cannot be set or does not exist, you may not have installed an interpreter, such as [Anaconda](https://www.anaconda.com/download/). Install Anaconda and restart Visual Studio Code before attempting to select the interpreter.

A notification will appear in the lower-right corner of the window, indicating that the Azure Machine Learning SDK is automatically being installed. The newly created Python environment is local and private and contains the Visual Studio Code prerequisites for working with the Azure Machine Learning service.

![A dialog appears stating that the Azure ML extension starting runtime dependencies...](media/vscode-azure-ml-extension-installing.png 'VS Code notification')

Follow the instructions [here](https://code.visualstudio.com/docs/python/python-tutorial) for configuring VS Code for Python development.

The AML extension adds a section to the Azure menu item on VS Code's left-hand menu (the extension adds the Azure menu item if it does not already exist) named _MACHINE LEARNING_. Within this section is a list of each of your Azure subscriptions. You can right-click on a subscription and add a new Azure Machine Learning service workspace through prompts in VS Code's command palette. When you expand an existing workspace, you can view experiments, pipelines, compute targets, models (each is versioned), deployed model images, and deployments.

The screenshot below shows the features for an AML workspace that have been expanded underneath a subscription in this new section. In this screenshot, a context menu displays after right-clicking on one of the models. You can see the following actions you can take on a model:

- Download Model file
- Remove Model
- View Model Properties
- Create Image From Model
- Deploy Service From Registered Model

![The screenshot displays an Azure Machine Learning service workspace that has been expanded within the new Machine Learning section. A context menu is shown by right-clicking on a model.](media/vs-code-aml-extension-workspace.png 'AML Extension workspace view')

## Visual interface

The Azure Machine Learning service visual interface can be used to prepare and visualize your data, run experiments, and quickly build, test, and deploy models through its easy-to-use visual drag-and-drop interface. If you are familiar with Azure Machine Learning Studio, you will see many similarities with Azure ML visual interface. Both services provide the same features using a nearly identical interface with low or node code, and Azure Machine Learning service visual interface is the successor to Azure Machine Learning Studio. However, there are some differences between both offerings. The two main advantages that visual interface provides are that you can scale your compute, and you can deploy your models to targets outside of Azure web services, such as Azure Kubernetes Service (AKS).

Here is a quick comparison of both options:

|                                                    | Machine Learning Studio                          | Azure Machine Learning service:<br/>Visual interface |
| -------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- |
| Modules for interface                              | Many                                             | Initial set of popular modules                       |
| Training compute targets                           | Proprietary compute target, CPU support only     | Supports Azure Machine Learning compute, GPU or CPU  |
| Deployment compute targets                         | Proprietary web service format, not customizable | Azure Kubernetes Service                             |
| Automated model training and hyperparameter tuning | No                                               | Not yet in visual interface                          |

It is important to note that any models created and deployed using visual interface can be managed through the Azure Machine Learning service workspace. This is another big advantage over Machine Learning Studio, where there are no options to export and use the models elsewhere.

To use Azure Machine Learning service visual interface, open your workspace in the Azure portal, then select the **Visual interface** link on the left-hand menu. This opens a blade that gives you the option to launch visual interface in a new browser window or read the documentation.

![Visual interface is launched by using the left-hand menu in your AML workspace.](media/visual-interface-link.png 'Visual interface')

The screenshot below shows an experiment in visual interface that contains steps to load and transform data, split the data for model training and evaluation, and scoring using the linear regression algorithm.

![This screenshot shows the UI for a visual interface experiment.](media/visual-interface-experiment.png 'Visual interface experiment')

When you are ready to run the experiment, you have the choice to select an existing compute target or create a new one. If you need more options for configuring the compute targets, such as VM size and type (CPU or GPU) or the number of nodes, you can create your custom compute targets within your workspace in the Azure portal.

When using a Machine Learning Compute target, there is a warmup time of approximately 5 minutes if you re-run your experiment after a long time. That is because the compute resource autoscales to 0 nodes when it is idle to save cost.

![The Setup Compute Target to Run Experiment dialog is displayed with Create new selected.](media/visual-interface-compute-target.png 'Setup compute target to run experiment')

After running the training experiment, you have an option to create a predictive experiment. When you create the predictive experiment, a new copy of your experiment is created that contains inputs and outputs so you can deploy your scoring model to a web service. A predictive experiment contains new options for defining inputs and outputs, and for deploying as a web service, as highlighted in the screenshot below. You can go back to the training experiment, make changes, and update the predictive experiment automatically with the click of a button.

![The predictive experiment is shown in a new tab and contains options for defining web service inputs and outputs.](media/visual-interface-predictive-experiment.png 'Predictive experiment')

After running the predictive experiment, you have the option to deploy a web service.

![The toolbar on the bottom of the page provides an option to deploy a web service.](media/visual-interface-deploy-web-service.png 'Deploy Web Service')

To deploy to a secure web service, we recommend adding a Kubernetes Service compute target to your workspace and [following the instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-secure-web-service) to use SSL to secure the web service. You can either select an existing Kubernetes Service or create a new one.

![The Compute menu item is selected in the AML workspace, and Kubernetes Service is selected as the compute type.](media/visual-interface-create-kubernetes-compute-target.png 'Add Compute')

After creating the compute target, you can select it from the list of existing targets.

![The newly created Kubernetes Service compute target is selected.](media/visual-interface-web-service-compute-target.png 'Setup Compute Target to Deploy Web Service')

After your web service deploys, you can view its details under Web Services on the visual interface site. The information provided includes the compute target and deployment state, and there are tabs that allow you to test your deployed model, sample code for consuming the web service, and an API Doc created from the generated Swagger file.

![The web service details includes sample code for consuming the service.](media/visual-interface-web-service.png 'Deployed web service')

Finally, you can go to your Azure Machine Learning service workspace and view the generated models, images, and deployments that were created by your visual interface experiment.

## Next steps

- [Configure your development environment](./environment-setup.md)

Read next: [Overview of Azure Machine Learning service architecture and concepts](./architecture-overview.md)
