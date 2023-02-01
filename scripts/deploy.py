#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to deploy productionalized model into AWS Sagemaker. The production model
from MLflow model registry in EC2 instance is packaged into a docker image as a
deployable model artifact and pushed into Amazon ECR. The deployable image from
AWS ECR is then deployed into AWS Sagemaker instance which creates an endpoint that
can be used to communicate with the model for inferencing.
"""

import os
import sys
import mlflow
from mlflow import sagemaker

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import Config

config = Config()

mlflow.set_tracking_uri(config["model-tracking"]["mlflow_tracking_uri"])

#Name of the resulting endpoint
app_name = config["model-deploy"]["endpoint_name"]

# Location of mlflow production model to be deployed from remote server
model_name = config["model-registry"]["model_name"]
model_uri = f"models:/{model_name}/production"

# Docker image that is built & pushed to AWS ECR repository as deployable model artifact
docker_image_url = os.environ["IMAGE_URI"]

# ARN role of IAM user
role = os.environ["ARN_ROLE"]

# Default region of AWS services
region = os.environ["REGION"]

# Deploying the docker image containing mlflow production model & dependencies from AWS ECR to Sagemaker instance
sagemaker._deploy(
                mode = 'create',
                app_name = app_name,
                model_uri = model_uri,
                image_url = docker_image_url,
                execution_role_arn = role,
                instance_type = 'ml.m5.xlarge',
                instance_count = 1,
                region_name = region
            )