# MLOps - DevOps for AI

DevOps has become ubiquitous in the world of classical development. Almost all projects that exceed a certain level of complexity become inevitably DevOps projects. Yet there is one category of projects that are stepping out of the line. You’ve guessed it right, it’s the category of Data Science projects.

When it comes to DevOps, Data Science projects pose a range of special challenges, whether it’s about the technical side of things, the philosophy of the people involved, or the actors involved. Think about one simple example: versioning. While in a “classical” development project versioning refers almost exclusively to source code, in the world of data science it gets another important aspect: data versioning. It’s not enough to know the version of the code for your model, it’s equally important to know the “version” of the data it was trained on. Another interesting question is, for examples, what does a “build” mean in the world of data science? Or a “release”?

This section is dedicated to the topic of MLOps which is basically the discipline that deals with the application of "classical" DevOps principles to projects that have at least one data science component. As the title implies, MLOps is `DevOps for AI`.

The following topics will be covered:

- [Overview of the relationship between model training pipeline (MLOps) and traditional DevOps application CI/CD pipelines](./mlops-and-devops/README.md)
- [Overview of post deployment monitoring and management tasks](./post-deployment-monitoring-and-management/README.md)
- [Monitoring a deployed model's collected data and telemetry (Code Sample)](./monitoring-data-and-telemetry-code-sample.md)
- [Model version management (Code Sample)](./model-version-management-code-sample.md)
- [Executing an end-to-end DevOps pipeline with Azure Machine Learning and Azure DevOps (Code Sample)](./e2e-pipeline-code-sample.md)

## Next steps

You can learn more about DevOps for AI by reviewing these links to additional resources:

- [MLOps: Manage, deploy, and monitor models with Azure Machine Learning Service](https://docs.microsoft.com/azure/machine-learning/service/concept-model-management-and-deployment)
- [Build reusable ML pipelines in Azure Machine Learning service](https://docs.microsoft.com/azure/machine-learning/service/concept-ml-pipelines)
- [Create and run a machine learning pipeline by using Azure Machine Learning SDK](https://docs.microsoft.com/azure/machine-learning/service/how-to-create-your-first-pipeline)
- [Train and deploy machine learning models](https://docs.microsoft.com/azure/devops/pipelines/targets/azure-machine-learning)

Read next: [Overview of the relationship between model training pipeline (MLOps) and traditional DevOps application CI/CD pipelines](./mlops-and-devops/README.md).
