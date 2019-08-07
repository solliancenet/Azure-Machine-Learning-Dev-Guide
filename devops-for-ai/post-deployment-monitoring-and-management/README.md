# Overview of post deployment monitoring and management tasks

When it comes to trained machine learning models, the typical data science project lifecycle includes the following stages:

- Initial model development
- Model training for production
- Trained model deployment into production
- Production use of the model
- Model retraining for production (a new cycle starts when the updated model is re-deployed into production)

Identifying the right moment to retrain a model for production is not a trivial task. Retraining too often means too many disruptions for systems that rely on the model and also potentially no significant improvements in performance. Retraining not often enough means potential degradation to the performance of the model. You will need to collect data for your production models in order to be able to perform the following tasks:

- Identify potential data drifts
- Correctly determine the right moment to retrain and/or optimize your model
- Incorporate monitoring data into the training dataset

This section addresses the problem of data collection and model versioning in post deployment (production) environments. The following topics will be covered:

- [Production monitoring of model input data and the deployed model predictions using the AML Model data collection](./monitoring-data-collection.md)
- [Collecting model web service performance telemetry (e.g., request rates, response times, failure rates and exceptions) with Application Insights](./model-webservice-performance-telemetry.md)

## Next steps

You can learn more about post depol by reviewing these links to additional resources:

- [Collect data for models in production](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-enable-data-collection)
- [Detect data drift (preview) on models deployed to Azure Kubernetes Service (AKS)](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-monitor-data-drift)
- [Monitor your Azure Machine Learning models with Application Insights](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-enable-app-insights)

Read next: [Production monitoring of model input data and the deployed model predictions using the AML Model data collection](./monitoring-data-collection.md)
