# Overview of Azure Machine Learning service architecture and concepts

Intro...

## High-level workflow

![The stages shown are Train, Package, Validate, Deploy, and Monitor. An arrow labeled Retrain goes back to Train from Monitor.](media/aml-model-workflow.png 'Azure Machine Learning model workflow')

### 1 - Train

At the core of the modern data science process is [training, evaluating, and selecting machine learning models](../modeling/feature-engineering-training-evaluation-selection/README.md). Once you have selected an algorithm for your model, you will need to train it with data that has been evaluated and prepared with the transformations and features required for training. At a very high level, training a model with Azure Machine Learning service involves the following steps:

- Using your favorite [Python environment](./environment-setup.md), create a machine learning training script along with any associated files. Specify the directory that contains these files, as well as an experiment name. Alternately, use visual interface for a code-free experience.
- Create and configure a compute target that will execute the training. During training, the entire directory is copied to the compute target before the training script is executed.
- Submit the training scripts to the configured compute target. The script will start running within the environment and has access to read from and write to [datastores](). Each execution saves a record of the run within the workspace, grouped under experiments.

Use the automated machine learning feature to automatically select the best model during training. This feature automates experimenting with multiple combination of parameter values, also referred to as hyperparameter tuning, to accelerate the model training process and keeps a record of the outcomes to help identify potential areas of improvement more quickly.

### 2 - Package

After you have trained your model and have identified the best version, you package it along with all the components you need to use the model, into an image. The image can be either a Docker image or an FPGA image used to deploy your model to a field-programmable gate array. The image is saved to the image registry in your workspace. The registry provides a centralized place to store your models so they can easily be copied to new deployment targets, as well as versioned.

### 3 - Validate

Model validation is used to calculate the accuracy of a model. Validation happens during the training process to make sure your chosen algorithm is performing as expected. It is also conducted periodically to ensure your model is still performing well over time with new data. Azure Machine Learning service allows you to query your experiments for logged metrics from current and past runs. Use the metrics to determine whether the run had a desired outcome. If not, begin the re-training process by starting over at step one.

### 4 - Deploy

Desc

### 5 - Monitor

Desc

### 6 - Re-deploy

Desc

## Components and concepts

Text

## Next steps

- [Reference link]()
- [Reference link]()

Read next: [Related article]()
