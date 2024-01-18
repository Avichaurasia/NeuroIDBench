# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

COPY requirements.txt ./

RUN pip install --upgrade pip

# Install the necessary packages to compile C extensions
# Install the necessary packages to compile C extensions
# Install the necessary packages to compile C extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    cython \
    pkg-config

# Install h5py as a binary wheel
RUN pip install --no-binary=h5py h5py

# Install any needed packages specified in requirements.txt
RUN pip install -r new_requirements.txt

# Run run.py when the container launches
CMD ["python", "brainModels/run.py"]
