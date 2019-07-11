# Overview of Feature Engineering, Model Training, Model Evaluation and Model Selection

Training, evaluating, and selecting the right Machine Learning models is at the core of each modern data science process. These processes are not possible though without another critical step: the correct preparation of data.  Correctly prepared data implies the following:

- Thorough understanding of the data - typically gained via a mix of business process knowledge and exploration.
- High quality of data - resulting from a combination of processes like cleaning, missing values handling, outlier detection, and transformation.
- Enrichment at feature level - a process which results in new features being derived from the original features available in the data set(s).

In the [Data acquisition & understanding](../../data-acquisition-understanding/README.md) section we have already discussed topics like [wrangling, exploring, and cleaning data](../../data-acquisition-understanding/data-wrangling.md). In this section we will focus on the following:


- [Feature Engineering introduced](./feature-engineering-introduced.md)
- [Model Training introduced](./model-training/README.md)
- [Model Evaluation introduced](./model-evaluation/README.md)

The main reason to further discuss Feature Engineering separately from the other data preparation tasks (and together with model training and evaluation) is that it is perhaps the one data preparation tasks that has the most significant impact on the potential success of a Machine Learning model.

## Next steps

You can learn more about feature engineering, model traning, and model evaluation by reviewing these links to additional resources:

- [Eplore and prepare data with the Dataset class](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-explore-prepare-data)
- [Transform data with the Azure Machine Learning Data Prep SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-transform-data)
- [Train and register Scikit-learn models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-scikit-learn)
- [Train and register Tensorflow models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-tensorflow)
- [Train and register Keras models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-keras)
- [Train and register PyTorch models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-pytorch)
- [Train models with Azure Machine Learning using estimator](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-ml-models)
- [Tune model hyperparameters](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters)

Read next: [Feature Engineering introduced](./feature-engineering-introduced.md)