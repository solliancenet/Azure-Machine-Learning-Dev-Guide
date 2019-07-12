# Introducing AML Estimators

An Estimator is an alternative higher-level abstraction used to construct run configurations when training deep learning models. For each training context, the Estimator contains the training code, the parameters, the compute resources specifications, and the runtime environment requirements. While the combination of `RunConfiguration` and `ScriptRunConfig` objects provides the highest level of control and flexibility, the `Estimator` object is designed to make the construction of run configurations easier.

Here is an example of using an Estimator in its simplest form:

```python
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
    '--regularization': 0.8
}

# run an experiment from the train.py code in your current directory
estimator = Estimator(source_directory='.',
                      script_params=script_params,
                      compute_target=<compute_target>,
                      entry_script='train.py',
                      conda_packages=['scikit-learn'])

# submit the experiment and then wait until complete
run = experiment.submit(estimator)
run.wait_for_completion()
```
Notice the parameters sent to the training script, the option to specify any `compute_target` (must be one that has already been created), and the requirement to install the `scikit-learn` Python package via Conda (there is an additional parameter available named `pip_packages` that can be used to specify packages to be installed via pip). 

## Parallel and distributed (multi-node) training

One particular limitation of the Estimator created in the code sample above is the fact that is runs in a single-node, non-parallel scenario. Consequently, it does not take advantage of the multi-node clusters that most Azure Machine Learning service compute targets can provide. The constructor of the `Estimator` object provides three additional paramters that control multi-node and parallel scenarios:

Name | Description
--- | ---
node_count | The number of nodes that will be used to run the training job. A value higher than 1 activates the distributed scenario. Only AML Compute targets are currently supported.
process_count_per_node | The number of worker processes that will run on each node. A value higher than 1 activates the parallel scenario. Note when this value is higher then 1, parallel execution will happen even if the value of `node_count` is 1. Only AML Compute targets are currently supported.
distributed_backend | When either `node_count` or `process_count_per_node` have a value higher than 1, must be set to `mpi` to instruct Azure Machine Learning to use its MPI implementation (based on [Open MPI](https://www.open-mpi.org/)).

The modified, multi-node and parallelized version of the Estimator will look like this:

```python
from azureml.train.estimator import Estimator

script_params = {
    '--data-folder': ds.as_mount(),
    '--regularization': 0.8
}

# run an experiment from the train.py code in your current directory
estimator = Estimator(source_directory='.',
                      script_params=script_params,
                      compute_target=<compute_target>,
                      entry_script='train.py',
                      node_count = 2,
                      process_count_per_node = 2,
                      distributed_backend = 'mpi',
                      conda_packages=['scikit-learn'])

# submit the experiment and then wait until complete
run = experiment.submit(estimator)
run.wait_for_completion()
```

## Using custom Docker images

One additional option available with Estimators is the possiblity of specifying your own Docker image as a base for building the execution environment. The constructor of the `Estimator` object provides additional paramters to configure the use of custom images. Out of these, the most important two are:

Name | Description
--- | ---
custom_docker_image | The name of the Docker image to be used. Must be an image from a public repository, like Docker Hub.
environment_definition | Used when an image from a private Docker repository must be used. Note that this parameter controlls also other settings, not related to the image to be used.

## Specialized Estimators

In addition to the general-purpose `Estimator` object, Azure Machine Learning provides the following specialized Estimators:

Name | Description
--- | ---
SKLearn | Used to launch a [Scikit-learn](https://scikit-learn.org/stable/index.html) training job on a compute target.
TensorFlow | Used to launch a [TensorFlow](https://www.tensorflow.org) training job on a compute target. If the `pip_packages` argument is used to import Keras on top of TensorFlow, this also can act as a [Keras](https://keras.io) Estimator.
PyTorch | Used to launch a [PyTorch](https://pytorch.org) training job on a compute target.
Chainer | Used to launch a [Chainer](https://chainer.org) training job on a compute target.

## Next steps

You can learn more about Estimators by reviewing these links to additional resources:

- [Train models with Azure Machine Learning using estimator](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-ml-models)
- [Train and register Scikit-learn models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-scikit-learn)
- [Train and register Tensorflow models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-tensorflow)
- [Train and register Keras models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-keras)
- [Train and register PyTorch models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-pytorch)
- [Train and register Chainer models at scale with Azure Machine Learning service](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-train-chainer)

Read next: [Introducing AML compute options (local and Azure Machine Learning compute)](./aml-compute-options.md)
