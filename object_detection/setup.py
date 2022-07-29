"""Setup script for object_detection with TF2.0."""
import os
from setuptools import find_packages
from setuptools import setup

# What packages are required for this module to be executed?
def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()


setup(
    name="object_detection",
    version="0.2",
    install_requires=list_reqs(),
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
