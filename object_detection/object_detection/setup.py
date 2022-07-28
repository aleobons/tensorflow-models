"""Setup script for object_detection with TF2.0."""
import os
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    # Required for apache-beam with PY3
    "avro-python3",
    "apache-beam",
    "pillow",
    "lxml",
    "matplotlib",
    "Cython",
    "contextlib2",
    "tf-slim",
    "six",
    "pycocotools",
    "lvis",
    "scipy",
    "pandas",
    "tf-models-official==2.7.2",
    "tensorflow-text==2.7.3",
    "tensorflow_io",
    "keras",
    "pyparsing==2.4.7",
    "tensorflow==2.7.0",
    "absl-py==1.2.0",
    "mlflow==1.27.0",
    "pydantic==1.9.1",
    "strictyaml==1.6.1",
]

setup(
    name="object_detection",
    version="0.2",
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=(
        [p for p in find_packages() if p.startswith("object_detection")]
        + find_packages(where=os.path.join(".", "slim"))
    ),
    package_dir={
        "datasets": os.path.join("slim", "datasets"),
        "nets": os.path.join("slim", "nets"),
        "preprocessing": os.path.join("slim", "preprocessing"),
        "deployment": os.path.join("slim", "deployment"),
        "scripts": os.path.join("slim", "scripts"),
    },
    description="Tensorflow Object Detection Library",
    python_requires=">3.6",
)
