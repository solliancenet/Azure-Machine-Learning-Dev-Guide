# What is automated machine learning (the concept)?

Developing a machine learning model usually involves multiple iterations using combinations of the following:

- Features available in the data sets
- Algorithms that are suitable for the task
- Parameter values that set the behavior of these algorithms (commonly referred as  hyper-parameters)
- Metrics used to measure model performance

The exploratory nature of machine learning development is very well described by the large number of these potential combinations. In fact, in many cases it is practically impossible to cover every single valid combination. Enters automated machine learning (simply referred as autoML).

The fundamental idea behind autoML is to enable the automated exploration of the above-mentioned combinations governed by an initial set of constraints defined as inputs for the process. Examples of such constraints include:

- The compute resources to be used.
- The machine learning prediction task type – currently only Classification, Regression, or Forecasting are supported.
- THe metric used for performance evaluation – like accuracy, spearman_correlation, normalized_root_mean_squared_error, and others.
- Whether preprocessing should be used or not.
- Exit criteria – limits imposed on the number of iterations, time of execution, and performance metric threshold.
- Model validation approach to be used.
- Concurrency levels.
- Algorithms to be included/excluded into the evaluation. 

These constraints enable you to control the output being produced as well as the resources being consumed to create that output.

## How does automML work?

Once you have specified the contraints that govern the automated exploration process, Azure Machine Learning autoML will create a number of parallel pipelines that will different models using different values for their hyper-parameters. Based on the performance evaluation metric that you specified, a ordered list of trained models is maintained and updated continuously (at a specified interval of time) during the exploratio process.

The following diagram describes at a conceptual level how autoML works:

![How does Automated Machine Learning work - high level](./media/automl-how-it-works-simple.png)

There are two ways to interact with Azure Machine Learning service autoML:

- Using the Azure Portal (user interface)
- Using the Azure Machine Learning service SDK for Python (code)

All trained models are serialized as `.pkl` files which can be later accessed through the SDK or even manually downloaded from the Azure Portal and used for scoring. They can also be deployed using the model deployment services provided by the Azure Machine Learning service.

The following diagram describes in more detail how autoML works:

![How does Automated Machine Learning work](./media/automl-how-it-works.png)

For each trained model, the service will log training information, including performance metrics. Read the [Measuring performance of classification and regression models](../feature-engineering-training-evaluation-selection/model-evaluation/measure-performance-regression-classification.md) section for a detailed discussion about performance metrics and charts produced by autoML.

## Next steps

You can learn more about Automated Machine Learning by reviewing these links to additional resources:

- [What is automated machine learning?](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml)
- [Understand automated machine learning results](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-understand-automated-ml)

Read next: [Capabilities of automated machine learning (the feature): feature engineering, algorithm selection, hyperparameter tuning, and model explainability](./capabilities-of-automated-machine-learning.md)