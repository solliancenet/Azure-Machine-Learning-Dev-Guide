# Azure Machine Learning developer guide

For years, Microsoft has been leading the way in AI and in providing the tools and services needed to help organizations of all sizes benefit from the capabilities it provides. Microsoft provides a vast number of technologies that you, as a data scientist, developer, or data engineer, can use to harness the power of AI and machine learning (ML). These technologies cover the [AI and ML spectrum](./intro/what-is-azure-machine-learning.md#the-microsoft-ai-and-ml-spectrum), from fully managed pre-built AI models ([Azure Cognitive Services](https://azure.microsoft.com/services/cognitive-services/)) to tools and services that enable you to build and deploy your custom models, with [Azure Machine Learning service](https://docs.microsoft.com/azure/machine-learning/service/overview-what-is-azure-ml) at its core.

This guide provides documentation, code samples, and best practices for harnessing the power of Azure Machine Learning. Whether you are a data scientist, data engineer, or developer, this guide is your resource to learn AI fundamentals and how to use Azure Machine Learning service and related tools to grow your AI practice and learn new skills.

Use the table of contents for diving into any part of the guide. You do not need to read the guide from front to back. It is divided into topics that introduce related concepts and contains detailed instructions and best practices. Each topic is self-contained and links to related content as needed.

## Table of contents

1. [Intro](./intro/README.md)
   - [What is Azure Machine Learning?](./intro/what-is-azure-machine-learning.md)
   - [What tools are used to do data engineering, data science, and AI?](./intro/tools.md)
   - [Overview of Azure Machine Learning service architecture and concepts](./intro/architecture-overview.md)
   - [Getting your environment set up](./intro/environment-setup.md)
2. [Data acquisition & understanding](./data-acquisition-understanding/README.md)
   - [Overview of wrangling, exploring and cleaning data](./data-acquisition-understanding/data-wrangling.md)
   - [Accessing data from Azure services and datastores](./data-acquisition-understanding/accessing-data.md)
   - [Load, transform and write data with Azure Machine Learning and the AML Data Prep SDK](./data-acquisition-understanding/loading-and-writing-data.md)
3. [Modeling](./modeling/README.md)
   - [Overview of Feature Engineering, Model Training, Model Evaluation and Model Selection](./modeling/feature-engineering-training-evaluation-selection.md)
     - [Feature Engineering introduced](./modeling/feature-engineering-training-evaluation-selection.md#feature-engineering-introduced)
     - [Model Training introduced](./modeling/model-training.md)
     - [Model Evaluation introduced](./modeling/model-evaluation.md)
   - [Training and Evaluating a simple model using Azure Machine Learning Visual Interface](./modeling/training-evaluating-model-with-visual-interface.md)
   - [Training and Evaluating a few simple models Using Azure Notebooks and Azure Machine Learning compute](./modeling/training-evaluating-simple-models-with-aml-compute.md)
   - [Simplifying the process with Automated Machine Learning, a component of Azure Machine Learning service](./modeling/simplify-process-with-automated-ml.md)
     - [Understanding automated machine learning generated models, using the model explainability capability of automated machine learning](./modeling/automl-understand-models-with-explainability.md)
     - [Using automated machine learning for Classification (Code Sample)](./modeling/automl-classification-code-sample.md)
     - [Using automated machine learning for Regression (Code Sample)](./modeling/automl-regression-code-sample.md)
     - [Using automated machine learning for Forecasting (Code Sample)](./modeling/automl-forecasting-code-sample.md)
     - [Using automated machine learning with model explainability (Code Sample)](./modeling/automl-understand-models-with-explainability.md#Model-explainability-code-sample)
4. [Model deployment](./model-deployment/README.md)
   - [Overview of deployment target options](./model-deployment/deployment-target-options.md)
   - [Overview of Real-time inferencing](./model-deployment/real-time-inferencing.md)
   - [Overview of Batch inferencing](./model-deployment/batch-inferencing.md)
   - [Overview of inferencing at the IoT edge](./model-deployment/iot-edge-inferencing.md)
   - [Reducing model deployment dependencies and improving model inferencing performance with ONNX](./model-deployment/deployment-with-onnx.md)
5. [Creating machine learning pipelines](./creating-machine-learning-pipelines/README.md)
6. [MLOps - DevOps for AI](./devops-for-ai/README.md)
   - [Overview of the relationship between model training pipeline (MLOps) and traditional DevOps application CI/CD pipelines](./devops-for-ai/mlops-and-devops.md)
   - [Overview of post deployment monitoring and management tasks](./devops-for-ai/post-deployment-monitoring-and-management.md)
   - [Monitoring a deployed model's collected data and telemetry (Code Sample)](./devops-for-ai/monitoring-data-and-telemetry-code-sample.md)
   - [Model version management (Code Sample)](./devops-for-ai/model-version-management-code-sample.md)
   - [Executing an end-to-end DevOps pipeline with Azure Machine Learning and Azure DevOps (Code Sample)](./devops-for-ai/e2e-pipeline-code-sample.md)
7. [Conclusion](./conclusion/README.md)
