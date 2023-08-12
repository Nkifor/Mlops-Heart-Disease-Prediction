import mlflow.deployments as mlflow_deployments

#import mlflow.sagemaker as mfs
import boto3

from markupsafe import escape
import yaml


session = boto3.Session(profile_name='prediction_instalation')
credentials = session.get_credentials()

# Przypisz dane uwierzytelniające do zmiennych
access_key = credentials.access_key
secret_key = credentials.secret_key
#aws_id = credentials.AWS_ID
#aws_arn = credentials.AWS_ARN

#token = credentials.token
print(f'Access key: {access_key}')
print(f'Secret key: {secret_key}')

#


#try:
#    with open("secret.yaml", "r") as f:
#        data = yaml.safe_load(f)
#        aws_id = data["AWS_ID"]
#        aws_arn= data["AWS_ARN"]
#except FileNotFoundError:
#    print("File 'secret.yaml' not found.")
#    exit(1)
#except Exception as e:
#    print(f"An error occurred while reading 'secret.yaml': {e}")
#    exit(1)

#print(aws_id)
#print(aws_arn)

target_uri = 'sagemaker:/eu-central-1/arn:aws:iam::325653208527:role/aws-sagemaker-for-deploy-ml-model'

client = mlflow_deployments.get_deploy_client(target_uri)




# Uzyskaj dane uwierzytelniające dla sesji
#credentials = session.get_credentials()
#
## Przypisz dane uwierzytelniające do zmiennych
#access_key = credentials.access_key
#secret_key = credentials.secret_key
##token = credentials.token
#print(access_key)
#print(secret_key)
#

experiment_id = '841523004778958357'
run_id = '60d6732f62f34285b65a6e689068efdb'
region = 'eu-central-1'
aws_id =  '325653208527'
#f'{aws_id}'
arn = 'arn:aws:iam::325653208527:role/aws-sagemaker-for-deploy-ml-model'
#f'{aws_arn}'
app_name = 'model-heart-application'
model_uri = f'mlruns/841523004778958357/60d6732f62f34285b65a6e689068efdb/artifacts/model'
# 841523004778958357 ? experiment_id
##f'runs:/{run_id}/model'
tag_id = '2.5.0'

image_url = aws_id + '.dkr.ecr.' + region + '.amazonaws.com/mlflow-pyfunc:' + tag_id
print(target_uri)
print(arn)
print(aws_id)
print(image_url)

config = {
    "region_name": region,
    "execution_role_arn": arn,
    "name": app_name,
    "instance_type": "ml.m5.2xlarge",
    "image_url": image_url,
    "bucket": "mlflow-model-heart-application",
}

try:
    client.create_deployment(name = app_name,
                             model_uri=model_uri,
                             config=config
           #region_name=region,
           #mode='create',
           #execution_role_arn=arn,
           )
except Exception as e:
    print(f"An error occurred while creating the deployment: {e}")
