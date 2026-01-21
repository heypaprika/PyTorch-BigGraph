export DOCKERHUB_USER="caleb70"
export IMAGE_NAME="memex-feed"
export GIT_SHA="$(git rev-parse --short HEAD)"

docker login

docker build -f Dockerfile.runpod -t "$DOCKERHUB_USER/$IMAGE_NAME:runpod-$GIT_SHA" .
docker tag "$DOCKERHUB_USER/$IMAGE_NAME:runpod-$GIT_SHA" "$DOCKERHUB_USER/$IMAGE_NAME:latest"

docker push "$DOCKERHUB_USER/$IMAGE_NAME:runpod-$GIT_SHA"
docker push "$DOCKERHUB_USER/$IMAGE_NAME:latest"