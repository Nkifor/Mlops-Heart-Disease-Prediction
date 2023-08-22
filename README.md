# Mlops Heart Disease Prediction

## Project description:

Project present implementation of Lasso Regression model including factors causing heart disease.
Model is trained, cross validated and by experimentation chosen the best one.
Selected model is deployed as Sagemaker endpoint.

## Main Technology used:


    ![Sagemaker](https://img.shields.io/badge/Sagemaker-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)
    ![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
    ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
    ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
    ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)




## Steps to reproduce project:
 1.  clone repo
 2.  create iam user name
 3.  create group of policies with full access to AmazonSageMaker and ECR and assign user to group
 4.  run console with command ```aws configure`` and provide information from user security credential:
    - access key,
    - secret key,
    - check default region of aws
    - default output format: json
 5.  check if mlflow user ui is shown correctly by typing: ```mlflow ui``` in console
 6.  run ```python app.py``` and check if model is shown correctly in mlflow ui(make sure that other mlflow session are closed)
 7.  create docker image in ECR by finding path of model in mflow and after setting this folder in terminal then run
     ```sagemaker build-and-push-container```
 8.  get your aws service id and save it to provide in deploy.py file by typing command
    ```aws sts get-caller-identity --query Account --output text```
 9. create role with policy AmazonSageMakerFullAccess and save arn_id to provide in deploy.py file from created role



## Data Labels and wider description:

- **age: age in years**
- **sex: sex**
    - 1 = male
    - 0 = female
- **cp: chest pain type**
    - Value 0: typical angina
    - Value 1: atypical angina
    - Value 2: non-anginal pain
    - Value 3: asymptomatic
- **trestbps: resting blood pressure (in mm Hg on admission to the hospital)**
- **chol: serum cholestoral in mg/dl**
- **fbs: (fasting blood sugar > 120 mg/dl)**
    - 1 = true;
    - 0 = false
- **restecg: resting electrocardiographic results**
    - Value 0: normal
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
- **thalach: maximum heart rate achieved**
- **exang: exercise induced angina**
    - 1 = yes
    - 0 = no
- **oldpeak = ST depression induced by exercise relative to rest**
- **slope: the slope of the peak exercise ST segment**
    - Value 0: upsloping
    - Value 1: flat
    - Value 2: downsloping
- **ca: number of major vessels (0-3) colored by flourosopy**
- **thal:**
    - 0 = error (in the original dataset 0 maps to NaN's)
    - 1 = fixed defect
    - 2 = normal
    - 3 = reversable defect
- **target (the lable):**
    - 0 = no disease,
    - 1 = disease



## Source of the data
[Source of the data](https://www.kaggle.com/code/desalegngeb/heart-disease-predictions/notebook).