# What is Azure Machine Learning?

Before we delve into the components of Azure Machine Learning service and its tools, let us first describe the landscape of artificial intelligence, what it means to you as a developer, data scientist, or data engineer, and then how Azure ML fits into the landscape and addresses your needs.

## What AI/deep learning/machine learning mean to the developer, data scientist, and data engineer

Artificial Intelligence is a term that has been around since the 1950s to describe a set of processes that enable computers to think more like humans and to learn on their own. Computers have long been used to solve problems, but the field of AI aims to have machines use information from the past and use that data to inform future decisions. Sometimes, this information from the past is created by the machine performing its own experiments and evaluating the outcomes. This is known as "reinforcement learning", which is a subdiscipline of machine learning that enables computers to do things like learning how to master chess, or how to drive a vehicle without crashing into obstacles. Machine learning is one example of a form of artificial intelligence that falls underneath the broader umbrella term of "AI".

In this article, we will focus on two primary AI components: **machine learning (ML)** and its unique subdiscipline, **deep learning**. As machine learning and deep learning become critical to the success of organizations, developers, data engineers, and data scientists are expected to expand their knowledge to meet these needs and achieve the requirements for digital transformation. AI becomes critical to an organization when the capabilities the technology provides is either at the core of that organization's business or enables the organization to innovate and gain a competitive edge in the marketplace. Let us continue by defining the primary fields of machine learning and deep learning.

### Machine learning

Machine learning is a data science technique used to extract predictions from statistical models by allowing computers to use existing data to forecast future outcomes, behaviors, and trends. This method of learning is accomplished without explicitly programming routines for computers to follow. There are too many variables to reliably account for every potential data point, logic flow, and decision to effectively program an application that is flexible enough to work with any given problem set and knowledge domain. Machine learning overcomes the terse and rigid constraints of explicitly programmed instructions by using special algorithms to find patterns and insights in a wide range of data. Machine learning systems harness this flexibility to use data from sources such as apps, sensors, historical data, networks, and devices to build its own logic to solve a problem or extract insight.

There are several subdisciplines of machine learning that focus on different sets of problems:

#### Supervised learning

Supervised learning means that you have access to data where the outputs are already known. You use this labeled data to teach the algorithm what conclusions to arrive to when you train your model. This means that the data should have target values that describe the prediction, such as whether a flight was delayed, or information defined that are essential data points that can be used to make that prediction. In the case of predicting flight delays, this could include the origin and destination airports, date, airline, weather conditions, and whether the flight was delayed. You are responsible for selecting these data points, otherwise known as features, choosing a suitable algorithm, using a portion of the historical data for training the model, and a portion to test the model with data it has not yet seen. The trained algorithm, or model, can then be used to make predictions on new data that contain the same features.

Refer to the [Azure Machine Learning Algorithm Cheat Sheet](https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-cheat-sheet) to view algorithms you can use to conduct supervised learning. All of the algorithms listed on the sheet, except for K-means clustering, are used in supervised learning, with the regression and classification categories of algorithms used most often.

#### Unsupervised learning

When you have data that is neither classified nor labeled, you could use an unsupervised algorithm to act on the information without guidance. The goal of the algorithm is to group data samples according to patterns in similarities and distances among them. This could mean grouping the data into clusters, as K-means does, or finding different ways of looking at complex data so that it appears more uncomplicated than in its unstructured form. Unlike supervised learning, you do not provide any prior teaching, which leaves the machine to find the hidden structure in unlabeled data by itself.

One example of unsupervised learning is used in healthcare. Analysts detect causality and identify correlations humans may miss by [inputting health data like blood pressure, heart rate, weight, and prescription data](http://people.csail.mit.edu/dsontag/courses/mlhc_summer18/day2/causal_inference.pdf) into an unsupervised learning algorithm.

#### Reinforcement learning

Reinforcement learning takes a more organic, almost human approach to learning. This class of algorithms interacts with a simulated or real environment to explore different strategies that result in a maximum reward. The choices are reinforced by either a favorable or unfavorable outcome in the form of a reward signal. Each choice the algorithm makes when it encounters a new data point is impacted by how great the reward was in its last decision. In effect, the algorithm continually modifies its strategy with the driving goal to achieve the highest reward.

Reinforcement learning is highly prevalent in robotics, where the set of sensor readings at one point in time is a data point the algorithm uses to choose the robot's next action. As mentioned earlier, you can also see reinforcement learning in action in self-driving vehicles or computers that teach themselves how to master games like chess and Go.

### Deep learning

Deep learning is technically a subdiscipline of machine learning, but its capabilities and how it approaches problem-solving are quite different. It uses what is called a neural network architecture that mimics how a brain works, rather than using traditional statistical frameworks. This architecture contains several stacked layers on top of each other, with higher levels of abstraction occurring within each layer. This layering is where the term "deep" comes from. The more layers that you use, the deeper the neural network architecture, and the more abstract interpretations of data can be made. This approach is especially useful when working with data without structured attributes or features, leaving it up to the algorithm to come up with its own interpretation of what the input represents. In comparison to most conventional machine learning techniques, deep learning requires massive amounts of compute power, more training time, and enormous datasets. Deep learning techniques have been around for many years, but only because of recent breakthroughs in the size of available datasets and computational resources has it become possible to apply them to hard, real-world problems.

Because reinforcement learning uses the same approach of mimicking the human brain, it can be placed under the deep learning umbrella. Some other examples of deep learning include speech recognition, image and object recognition, and Natural Language Processing (NLP). These capabilities are achieved through the use of neural network architectures, such as convolutional, recurrent neural networks, and multilayer perception.

## How the Azure Machine Learning service fits in the picture and addresses their needs

[Azure Machine Learning service](/azure/machine-learning/service/overview-what-is-azure-ml) is a fully managed cloud service used to train, deploy, and manage machine learning models at scale. It fully supports open-source technologies, so you can use tens of thousands of open-source Python packages such as TensorFlow, PyTorch, and scikit-learn. Rich tools are also available, such as [Azure notebooks](https://notebooks.azure.com/), [Jupyter notebooks](http://jupyter.org), or the [Azure Machine Learning for Visual Studio Code](https://aka.ms/vscodetoolsforai) extension to make it easy to explore and transform data, and then train and deploy models. Azure Machine Learning service includes features that automate model generation and tuning with ease, efficiency, and accuracy.

Use Azure Machine Learning service to train, deploy, and manage machine learning models using Python and CLI at cloud scale. For a low-code or no-code option, use the interactive, [visual interface](/azure/machine-learning/service/ui-quickstart-run-experiment) (preview) to easily and quickly build, test, and deploy models using pre-built machine learning algorithms.

![Azure Machine Learning helps you prepare your data, build and register your models, then easily deploy them.](media/steps-to-using-azureml.png 'Steps to using Azure ML')

## Next steps

- [Reference link]()
- [Reference link]()

Read next: [Related article]()
