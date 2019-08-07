# Overview of the relationship between model training pipeline (MLOps) and traditional DevOps application CI/CD pipelines

## MLOps - DevOps for AI overview

DevOps has become ubiquitous in the world of classical development. Almost all projects that exceed a certain level of complexity become inevitably DevOps projects. Yet there is one category of projects that are stepping out of the line. You’ve guessed it right, it’s the category of Data Science projects.

When it comes to DevOps, Data Science projects pose a range of special challenges, whether it’s about the technical side of things, the philosophy of the people involved, or the actors involved. Think about one simple example: versioning. While in a “classical” development project versioning refers almost exclusively to source code, in the world of data science it gets another important aspect: data versioning. It’s not enough to know the version of the code for your model, it’s equally important to know the “version” of the data it was trained on. Another interesting question is, for examples, what does a “build” mean in the world of data science? Or a “release”?

This section is dedicated to the topic of MLOps which is basically the discipline that deals with the application of "classical" DevOps principles to projects that have at least one data science component. As the title implies, MLOps is `DevOps for AI`.

## The relationship between MLOps and traditional DevOps application CI/CD pipelines

When it comes to DevOps principles, any non-trivial project that includes a data science component will need to take advantage of a combination of features from the following two major platforms:

- [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/)
- [Azure DevOps](https://azure.microsoft.com/en-us/services/devops)

The official definition of **Azure Machine Learning service** is:

> *Azure Machine Learning service provides a cloud-based environment you can use to prep data, train, test, deploy, manage, and track machine learning models. Start training on your local machine and then scale out to the cloud. The service fully supports open-source technologies such as PyTorch, TensorFlow, and scikit-learn and can be used for any kind of machine learning, from classical ml to deep learning, supervised and unsupervised learning.*

The official definition of **Azure DevOps** is:

> *Azure DevOps provides developer services to support teams to plan work, collaborate on code development, and build and deploy applications. Developers can work in the cloud using Azure DevOps Services or on-premises using Azure DevOps Server, formerly named Visual Studio Team Foundation Server (TFS).*

The DevOps mindset applied to data science solutions is commonly referred as Machine Learning Operations (MLOps). The most important aspects of MLOps are:

- Deploying Machine Learning projects from anywhere
- Monitoring ML solutions for both generic and ML-specific operational issues
- Capturing all data that is necessary for full traceability in the ML lifecycle
- Automating the end-to-end ML lifecycle using a combination of Azure Machine Learning service and Azure DevOps

## Two complementary platforms

If you plan on implementing an end-to-end DevOps approach for your project, you will need to use both platforms (Azure Machine Learning service and Azure DevOps). Although Azure DevOps is a native, all-purpose DevOps platform, it lacks specific features that are required in data science projects. At the same time, Azure Machine Learning service is a native, all-purpose Data Science platform, but it does not have some of the core DevOps features that you will most certainly need.

Because of the very specific DevOps-type needs that data science projects have, the Azure Machine Learning service has started to provide DevOps-like services centered around the concept of ML Pipelines and model management. ML Pipelines provide the backend for data scientists, data engineers, and IT professionals to collaborate of specific tasks like data preparation (e.g. transformation, normalization), model training and evaluation, and deployment. Because of the term `pipelines`, one might get the sense that ML Pipelines are somewhat of a competitor to Azure DevOps pipelines. In fact, the two of them are designed to work together and their functionalities are complementary. Model management enables you to register and track trained ML models based on their name and version.

ML Pipelines are described in the [Overview of machine learning pipelines using the Azure Machine Learning SDK](../../creating-machine-learning-pipelines/machine-learning-pipelines.md) section.

Model management is described in the [Model versioning with the AML model registry](../post-deployment-monitoring-and-management/model-versioning.md) section.

Next, we will explore the relationships that exist between the various features of Azure Machine Learning service and Azure DevOps. 

## Feature comparison

The following table provides details about how the most important MLOps concepts are interpreted by the Azure Machine Learning service and Azure DevOps.

MLOps concept | Azure Machine Learning service | Azure DevOps | Primary Choice
--- | --- | --- | ---
Boards | Not supported | Provides a set of services that enable you to plan, track, and discuss work across different teams involved in the project. | Azure DevOps
Code Repos | Not Supported | Provides a set of services that enable you to work with Git repos, and collaborate using advanced file management and pull requests. Integrates with boards, enabling you to link source code, branches, and pull requests with work items. | Azure DevOps
Artifacts Management | Provides extensive support for managing machine learning-specific artifacts like experiments, runs, models, compute resources, models, images, deployments, and activities. Also capable of handling various by-products of the ML processes like outputs, logs, images, performance metrics, and many more. | Provides generic support for build artifacts management as well as services for storing Maven, npm, NuGet, and Python packages together. | Azure Machine Learning service (Azure DevOps will be used to handle artifacts required by the CI/CD pipelines).
Pipelines | Provides services geared towards data preparation, model training and evaluation, and deployment. Enables optimization of ML workflows with simplicity, speed, portability, and reuse. Also enables explicit naming and versioning of data sources, inputs, and outputs. | Provides support for generic, all-purpose Continuous Integration / Continuous Delivery pipelines for building, testing, and deploying solutions. Works with any code repo provider, any language, any platform, and any cloud. | Azure DevOps (for high-level orchestration, coordination, build, and release using a CI/CD approach). Specific steps from the Azure DevOps CI/CD pipelines call into Azure Machine Learning service through the Python SDK (includes the launch of ML pipelines driven by the Azure Machine Learning service).
Data Versioning | Provides support through Azure ML Datasets. | Not supported | Azure Machine Learning service
Model Training, Deployment, and Monitoring | Provides extensive support for registering, tracking, packaging, debugging, validating, profiling, converting, and using ML models. Provides support for understanding what data is send to the model, and what predictions it returns. | Not supported | Azure Machine Learning service

Looking at this table it is easy to observe the complementary nature of Azure Machine Learning service and Azure DevOps in the context of MLOps. Furthermore, the integration between the two is simplified by the [Azure Machine Learning extension](https://marketplace.visualstudio.com/items?itemName=ms-air-aiagility.vss-services-azureml) from Azure DevOps which enables the following:

- Azure Machine Learning workspace selection when defining a service connection in Azure DevOps
- Azure DevOps release pipeline triggering by a trained model created in a training pipeline

## A typical scenario

The typical scenario is defined by the following:

- Uses a combination of Azure DevOps and Azure Machine Learning service
- Azure Machine Learning Datasets help track and version data
- Azure Machine Learning experiments and runs manage code, data, and compute used to train a model
- Azure Machine Learning model registry captures all the metadata around models
- Azure DevOps Boards are used to manage the agile development process
- Azure DevOps Code Repos are used to manage all source code, including code that controls and interacts with the Azure Machine Learning service (through the Python SDK)
- Azure DevOps release and build pipelines are used to implement the high-level processes. In turn, they rely on the Python SDK to interact with ML pipelines from Azure Machine Learning service
- When a data scientists checks in code into the repo, an Azure DevOps pipeline will start a training run. The results will be then inspected to assess the performance characteristics of the trained model. If the newly trained model passes the quality checks, a different (release) pipeline will be triggered to deploy the model as a service.

## Next steps

You can learn more about Azure Machine  by reviewing these links to additional resources:

- [MLOps: Manage, deploy, and monitor models with Azure Machine Learning Service](https://docs.microsoft.com/azure/machine-learning/service/concept-model-management-and-deployment)
- [Build reusable ML pipelines in Azure Machine Learning service](https://docs.microsoft.com/azure/machine-learning/service/concept-ml-pipelines)
- [Create and run a machine learning pipeline by using Azure Machine Learning SDK](https://docs.microsoft.com/azure/machine-learning/service/how-to-create-your-first-pipeline)
- [Train and deploy machine learning models](https://docs.microsoft.com/azure/devops/pipelines/targets/azure-machine-learning)

Read next: [Using Azure DevOps](./using-azure-devops.md)