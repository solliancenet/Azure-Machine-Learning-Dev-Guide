# Feature Engineering introduced

Feature Engineering is a process which results in new features being derived from the original features available in the data set(s). In most cases, enrichment is followed by a feature selection process aimed towards reducing the dimensionality of the training problem. Complementary to feature selection, dimensionality reduction algorithms can also be used to achieve this goal.

## Feature Engineering

The main purpose of the Feature Engineering process is to help increase the power of the machine learning algorithms. It does this by using existing features to derive new features that might prove more helpful to the model during the training process. 

From the point of view of the place where the process is taking place, we can distinguish two major approaches:
- Engineering features at the data source
- Engineering features in the model training context

The immediate question is: ``Where is the right place do perform feature engineering?``. The answer to it is that it depends on the method of feature calculation and the capabilities of the various platforms involved. For example, if the data source is a relational database, it makes more sense to calculate a new feature based on summarization (e.g. sum or average) at the database layer, as it most probably is better equipped for the task than a Python library like Pandas. On the other hand, if we’re talking about deriving features by feature learning, a specialized Python library will be the better choice.

This brings us to the next topic, the classification of feature engineering approaches. There are certainly many valid approaches, and some of the most popular ones are:

- Aggregation (count, sum, average, mean, median, and the like)
- Part-of (year of date, month of date, week of date, and the like)
- Binning (grouping entities into bins and then applying aggregations)
- Flagging (boolean conditions resulting in True of False)
- Frequency-based (calculating the frequencies of the levels of one or more categorical variables)
- Embedding (transforming one or more categorical or text features into a new set of features, possibly with a different cardinality)
- Deriving by example


### Engineeing features at the data source

### Engineering features in the model training context

## Feature Selection

## Dimensionality Reduction

## Next steps

You can learn more about feature engineering by reviewing these links to additional resources:

- [Feature engineering in data science](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/create-features)
- [Create features in SQL Server using SQL and Python](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/create-features-sql-server)
- [Create features in a Hadoop cluster using Hive queries](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/create-features-hive)
- [Explore and prepare data with the Dataset class](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-explore-prepare-data)
- [Transform data with the Azure Machine Learning Data Prep SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-transform-data)

Read next: [Model Training introduced](./model-training/README.md)
