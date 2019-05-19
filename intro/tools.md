# What tools are used to do data engineering, data science, and AI?

## Introducing the notebook paradigm

Executable, or interactive, notebooks have a long history in science and academia. Notebooks were traditionally provided by applications such as [MATLAB](https://www.mathworks.com/products/matlab.html) and [Wolfram Mathematica](https://www.wolfram.com/mathematica/) to help scientists, students, professors, and mathematicians create self-documenting notebooks that others can use to reproduce experiments. To accomplish this, notebooks contain a combination of runnable code, output, formatted text, and visualizations. Over the past several years, web-based interactive notebooks have gained popularity with data scientists and data engineers to conduct exploratory data analysis and model training using a number of languages, such as Python, Scala, SQL, R, and others. The most popular notebooks in use today are [Jupyter](http://jupyter.org/), [Databricks Notebooks](https://docs.azuredatabricks.net/user-guide/notebooks/index.html), [R Markdown](jhttp://rmarkdown.rstudio.com/), and [Apache Zeppelin](https://zeppelin.apache.org/).

Notebooks are made up of one or more of cells that allow for the execution of the code snippets or commands within those cells. They store commands and the results of running those commands. The image below shows a notebook that contains a cell on top that displays formatted text using markdown, followed by a cell containing Python code that gets executed to display an output. In this case, we chose a line chart visualization in place of the default text or table output. When the notebook is saved and shared with others, any outputs can be included so others can see the intended outcome of each cell.

![A screenshot of a notebook is displayed showing two cells. The first cell contains formatted text, and the second cell contains Python code and an output containing a line chart visualization.](media/notebook-cells.png 'Sample notebook cells')

Notebooks can be run locally, on a notebook server, such as [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/), or in a hosted environment, such as [Azure Notebooks](https://notebooks.azure.com/) or [Azure Databricks](https://docs.microsoft.com/en-us/azure/azure-databricks/what-is-azure-databricks). Depending on your notebook environment, you can connect to a number of data sources and select from a large collection of open source libraries and SDKs (such as the Azure Machine Learning SDK) supported by the notebook's kernel (Python, etc.) to accelerate your development efforts. When you execute the cells in your notebook, you must first connect to a local execution engine or a cluster. This gives you the flexibility to change the environment in which the embedded code can execute, allowing you to share the notebook across different environments and scale out computational workloads as needed. In most cases, you will run your notebooks in a cluster environment when performing model training. This allows you to execute more complex computational processing and use larger data sets than you would be able to from your own machine.

If you are used to developing software and applications using your favorite IDE, then you will realize that there are some disadvantages to using notebooks in place of a more traditional development platform. For example, you cannot set breakpoints and run in debug mode, allowing you to step through the code and inspect object and environment states during execution. However, there are many advantages notebooks do provide. They offer an environment that allows for exploration, documentation, collaboration, and visualization. When a data scientist creates and shares it with a colleague, they are sharing notes and insights about the data with access to all of the queries, formulas, visualizations, and models. This enables interactive conversations and further exploration, with simple reproducibility by anyone running the notebook in the same or similar environment, without others needing to know a sequence of shell commands and environment variables known only to the original author. This collaborative knowledge exchange within an easy to share self-contained package is far more valuable than simply sharing a static, final report.

## Azure Notebooks and Jupyter Notebooks

Text

## Visual Studio Code with the AML extension

Text

## Azure Machine Learning Studio

Text

## Next steps

- [Reference link]()
- [Reference link]()

Read next: [Related article]()
