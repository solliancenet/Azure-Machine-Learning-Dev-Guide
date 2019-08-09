# Introducing AML compute options (local and Azure Machine Learning compute)

An Azure Machine Learning Compute Target defines a compute resource you can use to:

- Run your model training script
- Host your service deployment (based on a trained model)

The compute resource identified by a Compute Target can be either local (the local computer) or remote. To manage Compute Targets, you can use one of the following:
- The Azure Portal
- The Azure Machine Learning SDK for Python
- The Azure Command Line Interface (CLI)
- The Visual Studio Code extension for Azure Machine Learning.

Throughout this article we will focus on the first two approaches (Azure Portal and SDK).

The following Compute Targets are supported for running model training scripts:

Name | Description
--- | ---
Local computer | Uses the resources provided by the local computer.
Azure Machine Learning Compute | A compute resource fully managed by Azure Machine Learning service, with support for CPU/GPU, single or multi-node clusters, autoscale, automatic cluster management, and job scheduling. Currently, this is the only managed compute resource available in Azure Machine Learning.
Virtual Machine | A remote Virtual Machine providing compute resources.
[Azure Databricks](https://azure.microsoft.com/services/databricks/) | An Azure Databricks (Apache Spark) cluster running in Azure.
[Azure Data Lake Analytics](https://azure.microsoft.com/services/data-lake-analytics/) | An Azure Data Lake Analytics big data analytics platform running in Azure.
[Azure HDInsight](https://azure.microsoft.com/services/hdinsight/) | A Hadoop big data cluster running in Azure.
[Azure Batch](https://azure.microsoft.com/services/batch/) | Environment for running large scale parallel HPC workloads in Azure.

The following Compute Targets are supported for hosting service deployments:

Name | Description
--- | ---
Local web service | Used for debugging, testing, and troubleshooting.
[Azure Kubernetes Service](https://docs.microsoft.com/azure/aks/intro-kubernetes) (AKS) | Used for production deployments, supports very fast response times and autoscaling. 
[Azure Container Instances](https://azure.microsoft.com/services/container-instances/) (ACI) | Used for testing and development, suitable for low scale CPU-based workloads.
Azure Machine Learning Compute | Used for batch inference.
[Azure IoT Edge](https://azure.microsoft.com/services/iot-edge/) | Used for deployment on IoT devices.
[Azure Data Box Edge](https://docs.microsoft.com/azure/databox-online/data-box-edge-overview) | Used for deployment on IoT devices.

For the reminder of this article we will focus on Compute Targets used to run model training scripts.

## Create or attach a Compute Target using the Azure portal

The Azure Portal provides support for viewing existing Compute Targets as well as creating and/or attaching new ones.

![View Azure Machine Learning Compute Targets in the Azure Portal](./media/compute-target-in-portal.png)

As mentioned above, the Azure Machine Learning Compute is the only Compute Target that is fully managed by the Azure Machine Learning service. Consequently, it is the only one that can be directly created from a workspace (all the others must be created separately and then attached - for details about setup requirements, read the [Configure a development environment for Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/service/how-to-configure-environment) article). 

This is an example of creating an Azure Machine Learning Compute cluster from the Azure Portal:

![Create a new Azure Machine Learning Compute cluster in Azure Portal](./media/aml-compute-create-in-portal.png)

## Create an Azure Machine Learning Compute cluster using the SDK

This is an example of creating a persistent Azure Machine Learning Compute cluster using the SDK:

```python
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=<cpu_cluster_name>)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, <cpu_cluster_name>, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
```
Note the Virtal Machine size being used (`STANDARD_D2_V2`) and the maximum number of nodes allowed in the cluster (4).

**Note**: A new option to create Azure Machine Learning Compute resources is being added to the service (currently in preview) - one that allows run-based creation. It uses the `RunConfiguration` object from the SDK to fully specify the requirements of the cluster. It also has some important limitations, namely it does not support automated hyperparameter tuning or automated machine learning. For mode details, read the [Set up compute targets for model training](https://docs.microsoft.com/azure/machine-learning/service/how-to-set-up-training-targets) article.

## Atach a Virtual Machine as a Compute Target using the SDK

This is an example on how to use the SDK to attach a Virtual Machine as a Compute Target:

```python
from azureml.core.compute import RemoteCompute, ComputeTarget

attach_config = RemoteCompute.attach_configuration(address = "<fqdn>",
                                                 ssh_port=22,
                                                 username='<username>',
                                                 password='<password>')

# Attach the compute
compute = ComputeTarget.attach(ws, <compute_target_name>, attach_config)

compute.wait_for_completion(show_output=True)
```

## Attach an Azure Databricks cluster as a Compute Target using the SDK

This an example on how to use the SDK to attach an Azure Databricks cluster as a Compute Target:

```python
from azureml.core.compute import ComputeTarget, DatabricksCompute
from azureml.exceptions import ComputeTargetException

try:
    databricks_compute = ComputeTarget(workspace=ws, name=databricks_compute_name)
    print('Compute target already exists')
except ComputeTargetException:
    print('compute not found')

    # Create attach config
    attach_config = DatabricksCompute.attach_configuration(resource_group = <databricks_resource_group>,
                                                           workspace_name = <databricks_workspace_name>,
                                                           access_token = <databricks_access_token>)
    databricks_compute = ComputeTarget.attach(
             ws,
             <databricks_compute_name>,
             attach_config
         )
    
    databricks_compute.wait_for_completion(True)
```

## Attach an Azure Data Lake Analytics platform as a Compute Target using the SDK

This an example on how to use the SDK to attach an Azure Data Lake Analytics platform as a Compute Target:

```python
from azureml.core.compute import ComputeTarget, AdlaCompute
from azureml.exceptions import ComputeTargetException

try:
    adla_compute = ComputeTarget(workspace=ws, name=adla_compute_name)
    print('Compute target already exists')
except ComputeTargetException:
    print('compute not found')

    # create attach config
    attach_config = AdlaCompute.attach_configuration(resource_group = <adla_resource_group>,
                                                     account_name = <adla_account_name>)
    # Attach ADLA
    adla_compute = ComputeTarget.attach(
             ws,
             <adla_compute_name>,
             attach_config
         )
    
    adla_compute.wait_for_completion(True)
```

## Attach an Azure HDInsight cluster as a Compute Target using the SDK

This an example on how to use the SDK to attach an Azure HDInsight cluster as a Compute Target:

```python
from azureml.core.compute import ComputeTarget, HDInsightCompute
from azureml.exceptions import ComputeTargetException

try:
    # if you want to connect using SSH key instead of username/password you can provide parameters private_key_file and private_key_passphrase
    attach_config = HDInsightCompute.attach_configuration(address='<clustername>-ssh.azureinsight.net', 
                                                        ssh_port=22, 
                                                        username='<ssh-username>', 
                                                        password='<ssh-pwd>')
    hdi_compute = ComputeTarget.attach(workspace=ws, 
                                        name=<hdinsight_compute_name>, 
                                        attach_configuration=attach_config)

except ComputeTargetException as e:
    print("Caught = {}".format(e.message))

hdi_compute.wait_for_completion(show_output=True)
```

## Attach an Azure Batch environment as a Compute Target using the SDK

This an example on how to use the SDK to attach an Azure Batch environment as a Compute Target:

```python
from azureml.core.compute import ComputeTarget, BatchCompute
from azureml.exceptions import ComputeTargetException

try:
    # check if the compute is already attached
    batch_compute = BatchCompute(ws, <batch_compute_name>)
except ComputeTargetException:
    print('Attaching Batch compute...')
    provisioning_config = BatchCompute.attach_configuration(resource_group=<batch_resource_group>, account_name=<batch_account_name>)
    batch_compute = ComputeTarget.attach(ws, <batch_compute_name>, <provisioning_config>)
    batch_compute.wait_for_completion()
```

## Next steps

You can learn more about compute options by reviewing these links to additional resources:

- [What are compute targets in Azure Machine Learning service?](https://docs.microsoft.com/azure/machine-learning/service/concept-compute-target)
- [Set up compute targets for model training](https://docs.microsoft.com/azure/machine-learning/service/how-to-set-up-training-targets)
- [Configure a development environment for Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/service/how-to-configure-environment)

Read next: [Introducing the AML Model Registry](./aml-model-registry.md)
