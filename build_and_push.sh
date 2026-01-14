#!/bin/bash

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-southeast-1 
REPO_NAME=memex-sagemaker
IMAGE_TAG=latest
DOCKERFILE=Dockerfile.sagemaker

# ECR repo create if not exists
aws ecr describe-repositories --repository-names "$REPO_NAME" > /dev/null 2>&1 || \
aws ecr create-repository --repository-name "$REPO_NAME"

# login, build, tag, and push the Docker image to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker build -f "$DOCKERFILE" -t "$REPO_NAME:$IMAGE_TAG" .
docker tag "$REPO_NAME:$IMAGE_TAG" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
