# TensorFlow Object Detection on Docker

These instructions are experimental.

## Building and running:

```bash
# From the root of the git repository
docker build -f object_detection/dockerfiles/tf2/Dockerfile -t aleobons/tensorflow_object_detection:0.2 .
docker run -it aleobons/tensorflow_object_detection:0.2
docker push aleobons/tensorflow_object_detection:0.2

```
