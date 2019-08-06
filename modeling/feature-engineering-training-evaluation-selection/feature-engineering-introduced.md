# Feature Engineering introduced

Feature Engineering is a process which results in new features being derived from the original features available in the data set(s). In most cases, enrichment is followed by a feature selection process aimed towards reducing the dimensionality of the training problem. Complementary to feature selection, dimensionality reduction algorithms can also be used to achieve this goal.

It is also important to note that Feature Engineering is not always required, depending on the specifics of the available data, the algorithm being used, and the objectives of the model being trained. 

## Feature Engineering

The main purpose of the Feature Engineering process is to help increase the power of the machine learning algorithms. It does this by using existing features to derive new features that might prove more helpful to the model during the training process. 

From the point of view of the place where the process is taking place, we can distinguish three major approaches:
- Engineering features at the data source
- Enginerring features in a dedicated data engineering environment (like a Spark cluster or an Azure Data Factory flow)
- Engineering features in the model training context

The immediate question is: ``Where is the right place to perform feature engineering?``. The answer to it is that it depends on the method of feature calculation and the capabilities of the various platforms involved. For example, if the data source is a relational database, it makes more sense to calculate a new feature based on summarization (e.g. sum or average) at the database layer, as it most probably is better equipped for the task than a Python library like Pandas. On the other hand, if we’re talking about deriving features by feature learning, a specialized Python library will be the better choice. This library can be run either in a dedicated environment for data engineering (like [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/)) or in the same context used for model training. Also, there is nothing preventing you from using any combination of the above mentioned options. In fact, modern data platforms are converging in terms of capabilities which opens a whole range of options. Take Python code for example. You can run it today directly in data sources (like [SQL Server 2017 and beyond](https://docs.microsoft.com/sql/advanced-analytics/tutorials/sql-server-python-tutorials?view=sql-server-2017)), in dedicated data engineering environment (like Azure Databricks), or in model training contexts (like Azure Machine Learning compute resources).

This brings us to the next topic, the classification of feature engineering approaches. There are certainly many valid approaches, and some of the most popular ones are:

- Aggregation (count, sum, average, mean, median, and the like)
- Part-of (year of date, month of date, week of date, and the like)
- Binning (grouping entities into bins and then applying aggregations)
- Flagging (boolean conditions resulting in True of False)
- Frequency-based (calculating the frequencies of the levels of one or more categorical variables)
- Embedding (transforming one or more categorical or text features into a new set of features, possibly with a different cardinality)
- Deriving by example

While there are many options of libraries to use for Feature Engineering tasks, the [Azure Machine Learning Data Prep SDK for Python](https://docs.microsoft.com/python/api/overview/azure/dataprep/intro?view=azure-dataprep-py) is a great option for both loading and transforming input data. The newer [Azure Machine Learning Datasets](https://docs.microsoft.com/azure/machine-learning/service/how-to-explore-prepare-data) library (currently in preview) is also very useful for data exploration and preparation.

### Engineering features at the data source

The typical scenario for engineering features at the data source is when data comes from a platform data has extended capabilities for indexing and querying. One such platform is a relational database/warehouse. In this case, it is quite common to perform data engineering tasks using the capabilities of the database. The two most widely used scenarios are:

- Calculate and save the engineered features to the database, then load them later into the model training context. This can be done using ETL (Extract-Transform-Load) processes that are either scheduled or run on-demand.
- Calculate the engineered features on-the-fly, while loading them into the model training context. This can be done using mechanisms like views or stored procedures.

To load data with engineered features, use the Azure Machine Learning Data Prep SDK.

```python
import azureml.dataprep as dprep

secret = dprep.register_secret(value="[SECRET-PASSWORD]", id="[SECRET-ID]")

ds = dprep.MSSQLDataSource(server_name="[SERVER-NAME]",
                           database_name="[DATABASE-NAME]",
                           user_name="[DATABASE-USERNAME]",
                           password=secret)
```
If the engineered features are already calculated, use a simple query to read them from the database.

```python
data = dprep.read_sql(ds, "SELECT * FROM [EngineeredFeaturesTable]")
```

If calculation happens in real time, call the relevant view or stored procedure.

```python
data = dprep.read_sql(ds, "SELECT * FROM [vEngineeredFeatures]")
```

### Engineering features in a dedicated data engineering environment

When the complexity of your data preparation process exceeds the capabilities of your individual data sources, you will most likely decide on using a dedicated environment for the process. One of example of such an environment is a Hadoop based one, either “classical” map/reduce (like [Azure HDInsight]( https://azure.microsoft.com/en-us/services/hdinsight/)) or Spark-based (like [Azure Databricks]( https://azure.microsoft.com/en-us/services/databricks/)). Using such an environment is especially useful when large volumes of heterogenous data are involved that need massive processing power.
These environments typically have support from multiple languages, including `Scala`, `SQL`, and `Python`. This enables you to take advantage of a very large array of options and libraries written for these languages. Also, the sheer computing power provided by the underlying clusters allow you to implement even to most complex and resource-demanding Feature Engineering tasks.
Extensive support for Python also means that you can reuse code and be flexible when it comes to deciding where exactly it should be executed.

For example, to load data from a SQL Azure database using Python, you can use a library like pyodbc.
```python
#Set up the SQL Azure connection
import pyodbc
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=<servername>;DATABASE=<dbname>;UID=<username>;PWD=<password>')

# Query database and load the returned results in pandas data frame
data_frame = pd.read_sql('''select <columnname1>, <columnname2>... from <tablename>''', conn)
```
Once you load your data into a [Pandas]( https://pandas.pydata.org/) data frame, you can go ahead and use a wide range of Python data science libraries to implement your Feature Engineering processes.


### Engineering features in the model training context

When using Azure Machine Learning, the model training context is provided by the compute target of your choice. Azure Machine Learning provides a wide range of choices, like your local computer, dedicated compute, remote virtual machines, Azure Databricks, Azure Data Lake Analytics, Azure HDInsight, or Azure Batch.

Since the model training context is essentially a Python environment, you have access to all Python data science libraries to implement your Feature Engineering tasks. Depending on the specific compute target being used, additional functionalities are also available (like Spark on Azure Databricks, for example).

Due to clarity, consistency, and traceability reasons, it is quite common to implement Feature Engineering tasks as part of the Python codebase that performs the actual model training.


## Why Feature Engineering?

The most effective way to understand the added value of Feature Engineering is through a practical example. Let’s take for example the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) which is based on real data from the Capital Bikeshare company that maintains a bike rental network in Washington DC in the United States. The dataset represents the number of bike rentals within a specific hour of a day in the years 2011 and year 2012 and contains 17389 rows and 17 columns. The raw feature set contains weather conditions (temperature/humidity/wind speed) and the type of the day (holiday/weekday). The field to predict is the `cnt` count, which represents the bike rentals within a specific hour and which ranges from 1 to 977.

For the sake of simplicity, let’s assume we want to build a regression model `Model1` to predict `cnt` using a feature set A that contains the `weather`, `holiday`, `weekday`, and `weekend` features which are originally available in the data set. Using the same algorithm, we build a new model (`Model2`) which uses the feature set A and an extra engineered feature set B which contains the number of bikes that where rented in each of the previous 12 hours. Next, we build a third model (`Model3`) which uses the feature sets A and B and an extra engineered feature set C which contains the number of bikes that were rented in each of the previous 12 days at the same hour. Finally, we build a fourth model (`Model4`) which uses the feature sets A, B, and C and an extra engineered feature set D which contains the number of bikes that were rented in each of the previous 12 weeks at the same hour and the same day.

While the details of training the models are outside the scope of our discussion, the most likely results obtained in terms of model performance will look like this (exact numbers may vary - for more details see Example 1 in [Feature engineering in data science](https://docs.microsoft.com/azure/machine-learning/team-data-science-process/create-features)):

Model | Features | MAE (Mean Absolute Error) | RMSE (Root Mean Square Error)
--- | --- | --- | ---
Model1 | A | 89.7 | 124.9 
Model2 | A + B | 51.7 | 88.3
Model3 | A + B + C | 47.6 | 81.1
Model4 | A + B + C + D | 48.3 | 82.1

This example demonstrates two remarkable aspects of Feature Engineering. The first one is the advantage gained by adding engineered features. `Model3` is the most effective one, and this is clearly due to the extra sets of features (B and C) which capture in an explicit way a reality that is “hidden” by the original features. The second one is what you could call “the reverse of the medal”. Despite the fact the `Model4` uses even more engineered features, is does not exhibit an improvement over `Model3`. On the contrary, it is slightly less powerful then `Model3`. Which demonstrates that avoiding abuse of Feature Engineering is also something to be seriously considered.


## Feature Selection

As you can easily see from the example in the section above, the Feature Engineering process results in a significant increase in the number of features of your data set(s). The obvious question is `Which are the features that are most useful for a given model?` Even if no new features are engineered at all, you may still find yourself in a situation where you need to decide on which features will be made available to a machine learning algorithm.

The process of filtering the features is commonly referred as Feature Selection. Feature Selection is a very important preparation step to ensure your machine learning model yields the best performance possible. There are two main reasons for using Feature Selection:

- Elimination of irrelevant, redundant, or (highly) correlated features. For example, in the case of a clustering algorithm like K-means, using simultaneously two features that have a significant statistical correlation score yields no additional value. On the contrary, increases the complexity of the model training process and affects performance.
- Reduction of the dimensionality of the training problem while also increasing the performance of model training process.
 
## Dimensionality Reduction

In many cases, the data sets being used can have many original features. If Feature Engineering is used, this number further increases, sometimes quite significantly. Using many features poses the following challenges:

- Some algorithms cannot cope with too many features at all.
- Some algorithms can cope with many features, but the performance and compute resource requirements increase dramatically as more features are taken into consideration.
- The explainability and/or understandability of the resulting trained model is very difficult when many features are used.

These challenges are commonly referred as “the curse of dimensionality”. In order to address them, several techniques can be deployed. Some of the most common are:

- PCA (Principal Component Analysis) – a linear dimensionality reduction technique based mostly on exact mathematical calculations.
- t-SNE (T-Distributed Stochastic Neighboring Entities) - a dimensionality reduction technique based on a probabilistic approach. Ideally suited for cases where the target number of dimensions is 2 or 3. 
- Feature embedding – a class of dimensionality reduction techniques based on using machine learning models to “encode” a larger number of features into a smaller number of features (sometimes also referred as “super-features”). Commonly used to avoid problems with categorical features, especially the ones with high cardinality.

Since the results of dimensionality reduction techniques are essentially new features, they are commonly considered to be a part of the larger family of Feature Engineering techniques. They are also considered to be a part of the Feature Learning class of techniques.

## Next steps

You can learn more about feature engineering by reviewing these links to additional resources:

- [Feature engineering in data science](https://docs.microsoft.com/azure/machine-learning/team-data-science-process/create-features)
- [Create features in SQL Server using SQL and Python](https://docs.microsoft.com/azure/machine-learning/team-data-science-process/create-features-sql-server)
- [Create features in a Hadoop cluster using Hive queries](https://docs.microsoft.com/azure/machine-learning/team-data-science-process/create-features-hive)
- [Explore and prepare data with the Dataset class](https://docs.microsoft.com/azure/machine-learning/service/how-to-explore-prepare-data)
- [Load and read data with the Azure Machine Learning Data Prep SDK](https://docs.microsoft.com/azure/machine-learning/service/how-to-load-data)
- [Transform data with the Azure Machine Learning Data Prep SDK](https://docs.microsoft.com/azure/machine-learning/service/how-to-transform-data)

Read next: [Model Training introduced](./model-training/README.md)
