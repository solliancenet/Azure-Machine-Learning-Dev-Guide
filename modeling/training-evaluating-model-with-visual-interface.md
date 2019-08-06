# Training and Evaluating a simple model using Azure Machine Learning Visual Interface

Azure Machine Learning Visual Interface gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning Visual Interface also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications. To use Azure Machine Learning Studio, you do not need programming experience and is designed for simplicity and productivity. The no code experience is most suited for:

- Data scientists who are not familiar with coding
- Users that are new to machine learning
- Rapid prototyping for experienced machine learning users

You can access Visual Interface from your machine learning workspace in Azure portal by selecting **Visual interface** in the left navigation and then select **Launch visual interface**.

![Steps to launch Visual Interface from your machine learning workspace in Azure portal](media/vi_launch.png 'Azure Machine Learning Visual Interface')

An experiment in Visual Interface comprises of datasets and analytical modules. A dataset is data uploaded to use in the modeling process and a module is an algorithm that you can perform on the data. Visual Interface provides rich set of modules for data preparation, feature engineering, training algorithms, and model evaluation. An experiment is created, using a drag-and-drop workflow, using datasets and modules which you connect together to construct a predictive model.

   ![Example experiment in Azure Machine Learning Visual Interface](media/vi_overview.png 'Azure Machine Learning Visual Interface - Experiment')

In this article we will first look at the list of analytical modules available in Visual Interface, and then look at examples of how to use these modules to do data prep, model training, and model evaluation.

## Analytical Modules Overview

A module represents algorithm or set of code you can run on the data. The Visual Interface provides a number of modules that range from data ingestion, data transformation, to model training, scoring and validation. This section summarizes the list of key modules in Visual Interface.

### Data Prep Modules

The Data Prep modules can be grouped into four types:

- Data format conversions
  - Convert a dataset into a CSV format that be downloaded, exported, or used with R or Python script modules.
- Data input and output
  - Use cloud storage to exchange data with the experiment.
- Data transformation
  - Data transformation modules that support various machine learning specific operations like normalizing, binning, feature selection or dimensionality reduction.
- Custom modules
  - Embed custom code written in Python or R in your experiment.

![Overview of Data Prep Modules](media/vi_dp_modules.png 'Data Prep Modules')

### Machine Learning Algorithms

The Machine Learning Algorithms modules can be grouped into three types:

- Classification Algorithms
  - Binary or multi-class classification algorithms.
- Regression Algorithms
  - Predict output values based in input features.
- Clustering Algorithms
  - Partition n observations into k clusters.

![Overview of Machine Learning Algorithms Modules](media/vi_ml_modules.png 'Machine Learning Algorithms Modules')

### Model Training and Evaluation

The Model Training and Evaluation modules can be grouped into three types:

- Model Training
  - Fit the model to the training data.
- Model Evaluation
  - Evaluate performance metrics for the trained model.
- Model Predictions
  - Make predictions using the trained model.

![Overview of Model Training and Evaluation Modules](media/vi_traineval_modules.png 'Model Training and Evaluation Modules')

## Dataprep Module Example

In this example, we are building a regression model to predict NYC Taxi Fares. We have created a new experiment in Visual Interface and we have already added the training dataset. Next, we want to preprocess the training data to address potential missing values in the input features. The steps involved in cleaning missing data are as follows:

1. Select the `Cleaning Missing Data` module from the left navigation under `Data Transformation -> Manipulation` section
2. Drag and drop the `Cleaning Missing Data` module onto the canvas
3. Connect the `Dataset` module to `Cleaning Missing Data` module
4. Select the `Cleaning Missing Data` module
5. Choose the desired `Cleaning mode`

The `Cleaning Missing Data` module supports several options for dealing with missing data such as, replace with mean, replace with median, replace with mode, remove entire row, remove entire column, and custom substitution value.

![Step by step instructions on how to configure the Cleaning Missing Data module](media/vi_cmd.png 'Cleaning Missing Data Module')

## Model Training Module Example

In order to setup model training, let's first pick our machine learning algorithm. In this example, we will use the `Boosted Decision Tree Regressor` algorithm.

1. Select the `Boosted Decision Tree Regressor` module from the left navigation under `Machine Learning -> Initialize Model -> Regression` section
2. Drag and drop the `Boosted Decision Tree Regressor` module onto the canvas
3. Select the `Boosted Decision Tree Regressor` module
4. Configure the model parameters, for example, change the `Learning rate` to 0.1

![Step by step instructions on how to configure the Boosted Decision Tree Regressor module](media/vi_bdtr.png 'Boosted Decision Tree Regressor Module')

Next, we will setup the `Train Model` module.

1. Select the `Train Model` module from the left navigation under `Machine Learning -> Train` section
2. Drag and drop the `Train Model` module onto the canvas
3. Connect the `Boosted Decision Tree Regressor` module to the `Train Model` module
4. Connect the first output of `Split Data` module to `Train Model` module
5. Select the `Train Model` module
6. Select `Edit columns` to setup the `Target` column

![Step by step instructions on how to configure the Train Model module](media/vi_tm.png 'Train Model Module')

Note that `Split Data` module is used to split the data for training and scoring. The first output will connect with the `Train Model` module and the second output will connect with the `Score Model` module.

In this example, we are predicting NYC Taxi Fares and our target column name is `totalAmount`. From the popup dialog, provide the name for the target column, and then select `Ok`.

![Step by step instructions on how to setup the target column](media/vi_tc.png 'Setup Target Column')

## Model Scoring and Evaluation Modules Example

The `Score Model` module is used to generate predictions using the trained model, whereas, the `Evaluate Model` module is used to generate performance metrics of the trained model. In this section, we will review how to setup the `Score Model` and the  `Evaluate Model` modules. In the next section, we will review the outputs from these two modules.

1. Select the `Score Model` module from the left navigation under `Machine Learning -> Score` section
2. Drag and drop the `Score Model` module onto the canvas
3. Connect the `Train Model` module to the `Score Model` module
4. Connect the second output of `Split Data` module to `Score Model` module
5. Select the `Evaluate Model` module from the left navigation under `Machine Learning -> Evaluate` section
6. Drag and drop the `Evaluate Model` module onto the canvas
7. Connect the `Score Model` module to the `Evaluate Model` module

![Step by step instructions on how to configure the Score Model and Evaluate Model modules](media/vi_sem.png 'Score Model and Evaluate Model Modules')

## Model Evaluation

To run the experiment that will train and evaluate the model, you select `Run` in the bottom bar menu in the experiment. You can either create a new machine learning compute from Visual Interface or choose an existing compute to run the experiment. Once the experiment run is completed successfully, then you can view the results from the `Score Model` module and the `Evaluate Model` module.

The score, or the predicted value, depends on the type of the model and the input data:

- Classification models: Output is the predicted value for the class and its associated probability.
- Regression models: The generated output is the predicted numeric value.
- Image classification models: Output is either the class of object in the image, or a Boolean indicating if a specific feature was found in the image.

To visualize results from the `Score Model` module, right-click on the module and then select `Scored dataset -> Visualize`. For the example of NYC Taxi Fare predictions, you can review the scored labels (predicted fare) and the associated histogram.

![Sample results from the Score Model module](media/vi_smv2.png 'Score Model Results')

The evaluation metrics that are generated by the `Evaluate Model` module are specific to the type of the machine learning model.

- Evaluation metrics for classification models include: Accuracy, Precision, Recall, F-score, AUC, Average log loss, and Training log loss.

- Evaluation metrics for regression models include: Mean absolute error (MAE), Root mean squared error (RMSE), Relative absolute error (RAE), Relative squared error (RSE), Mean Zero One Error (MZOE), and Coefficient of determination.

Please refer to the section on [Measuring performance of classification and regression models](../feature-engineering-training-evaluation-selection/model-evaluation/measure-performance-regression-classification.md) to learn more about the evaluation metrics for different types of machine learning problems.

To visualize results from the `Evaluate Model` module, right-click on the module and then select `Evaluation results -> Visualize`. You will observe a set of evaluation metrics that are relevant to the type of machine learning algorithm used. The NYC Taxi Fare predictor is a regression problem, and thus you will observe metrics such as, `Mean Absolute Error`, `Relative Squared Error`, `Root Mean Squared Error` etc.

![Sample results from the Evaluate Model module](media/vi_emv.png 'Evaluate Model Results')

## Next steps

Please see the following additional references on Azure Machine Learning Visual Interface:

- [Algorithm & module reference overview](https://docs.microsoft.com/azure/machine-learning/algorithm-module-reference/module-reference)
- [Visual interface for Azure Machine Learning service](https://azure.microsoft.com/en-us/blog/visual-interface-for-azure-machine-learning-service/)
- [What is Azure Machine Learning Studio?](https://docs.microsoft.com/azure/machine-learning/studio/what-is-ml-studio)

Read next: [Training and Evaluating a few simple models Using Azure Notebooks and Azure Machine Learning compute (Code Sample)](../training-evaluating-simple-models-with-aml-compute/README.md)
