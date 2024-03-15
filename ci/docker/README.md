# Build image with quiccir dependencies
docker build -f ci/docker/Dockerfile -t giacasti/ubuntu-llvm:22.04-17 .
docker push giacasti/ubuntu-llvm:22.04-17