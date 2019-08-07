# Collecting model web service performance telemetry (e.g., request rates, response times, failure rates and exceptions) with Application Insights

Azure Machine Learning is integrated with [Azure Application Insights](https://docs.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview).

The official definition of Azure Application Insights is:

>*Application Insights is an extensible Application Performance Management (APM) service for web developers on multiple platforms. Use it to monitor your live web application. It will automatically detect performance anomalies. It includes powerful analytics tools to help you diagnose issues and to understand what users actually do with your app. It's designed to help you continuously improve performance and usability. It works for apps on a wide variety of platforms including .NET, Node.js and Java EE, hosted on-premises, hybrid, or any public cloud. It integrates with your DevOps process, and has connection points to a variety of development tools. It can monitor and analyze telemetry from mobile apps by integrating with Visual Studio App Center.*

Enabling Application Insights telemetry on your production model provides the following benefits:

- Accurate measurement of request rates, response times, and failure rates
- Traceability for all usage
- History of exceptions

## Enabling Application Insights integration

You can use the Azure portal to enable Application Insights integration for a model that has been published in production:

![Enable Application Insights integration in Azure Machine Learning](./media/azureml-telemetry-enable.png)

Deploying a trained model to production is described in the [Overview of deployment target options](../../model-deployment/deployment-target-options.md) section.

Enabling data collection using the SDK is described in the [Monitoring a deployed model's collected data and telemetry (Code Sample)](../monitoring-data-and-telemetry-code-sample.md) section.

## Querying and analyzing monitoring data

You can use the Azure Portal to view and analyze the telemetry data collected by Application Insights:

![View Application Insights telemetry data in Azure Portal](./media/azureml-telemetry-azure-portal.png)

If you prefer to use Power BI to analyze telemetry data, you can do this using one of the following approaches:

- [Export Analytics queries](https://docs.microsoft.com/en-us/azure/azure-monitor/app/export-power-bi#export-analytics-queries)
- [Continuous export and Azure Stream Analytics](https://docs.microsoft.com/en-us/azure/azure-monitor/app/export-stream-analytics)

## Next steps

You can learn more about production model telemetry with Azure Application Insights by reviewing these links to additional resources:

-[Monitor your Azure Machine Learning models with Application Insights](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-enable-app-insights)

Read next: [Monitoring a deployed model's collected data and telemetry (Code Sample)](../monitoring-data-and-telemetry-code-sample.md)