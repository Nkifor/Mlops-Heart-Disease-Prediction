import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import json
import boto3
import logging
import mlflow.deployments as mlflow_deployments
from io import StringIO

global app_name
global region
app_name = 'model-heart-application'
region = 'eu-central-1'


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


with open("config.yaml", "r") as f:
    data = yaml.safe_load(f)
    data_path = data["data_source"]



try:
        data_frame = pd.read_csv(data_path, sep=",")
except Exception as e:
        logger.exception(
            "Unable to get training & test CSV, check your internet connection. Error: %s", e
        )

target_uri = "sagemaker"
#'sagemaker:/eu-central-1/arn:aws:iam::325653208527:role/aws-sagemaker-for-deploy-ml-model'



client = mlflow_deployments.get_deploy_client(target_uri)


def check_status(app_name):
    sage_client = boto3.client('sagemaker', region_name=region)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description['EndpointStatus']
    return endpoint_status


#def query_endpoint(app_name, input_json):
#    client = boto3.session.Session().client('sagemaker-runtime', region)
#
#    response = client.invoke_endpoint(
#        EndpointName = app_name,
#        Body = input_json,
#        ContentType = 'application/json; format=pandas-split',
#        )
#
#    preds = response['Body'].read().decode('ascii')
#    preds = json.loads(preds)
#    print('Received response: {}'.format(preds))
#    return preds


# Check endpoint status
print('Application status is {}'.format(check_status(app_name)))

# Prepare to give for predictions

print(data_frame)

train, test = train_test_split(data_frame)



train_x = train.drop(["target"], axis=1)
test_x = test.drop(["target"], axis=1)
train_y = train[["target"]]
test_y = test[["target"]]

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

## Create test data and make inference from endpoint
#query_input = test_x.iloc[[15]].to_json(orient='split')
csv_buffer = StringIO()
df_filtered = test_x.head(5)
df_filtered.to_csv(csv_buffer, header=False, index=False)
df_converted_to_csv_data = csv_buffer.getvalue()



#df_filtered = test_x.head(5)
#df_filtered.columns = df_filtered.iloc[0,:].values
#df_filtered_no_header = df_filtered.tail(-1)
#print(df_filtered_no_header)
#print(type(df_filtered_no_header))

#dict_data = df_filtered_no_header.to_dict(orient='records')
#
#
#input_data = {
#    "instances": dict_data
#}
#
#
#input_json = json.dumps(input_data)

df_converted_to_csv_data
#
sage_client = boto3.client('sagemaker-runtime', region_name=region)


response = sage_client.invoke_endpoint(
    EndpointName=app_name,
    Body=df_converted_to_csv_data,
    ContentType='text/csv'
    #"application/json"
)


print(df_converted_to_csv_data)

response_body = response['Body'].read().decode('utf-8')

#result = json.loads(response_body)
print('Result is'+ response_body)

#print(query_input)

# Invoke the deployed model
#result = client.predict(app_name, input_json)
#print(result)

#predictions = query_endpoint(app_name=app_name, input_json=query_input)