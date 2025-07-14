#!/bin/bash
set -e  # Exit on error

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=ap-northeast-1 
REPO_NAME=pbg-sagemaker
IMAGE_TAG=latest
PLATFORM=linux/amd64

# ECR repo create if not exists
aws ecr describe-repositories --repository-names "$REPO_NAME" > /dev/null 2>&1 || \
aws ecr create-repository --repository-name "$REPO_NAME" --region "$REGION"

# ECR login
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin \
  "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# image build
docker buildx build --platform "$PLATFORM" -t "$REPO_NAME:$IMAGE_TAG" . --load

# tag the image
docker tag "$REPO_NAME:$IMAGE_TAG" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"

# push the image to ECR
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:$IMAGE_TAG"
