FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install -r requirements.txt

# Set the working directory back to /workspace
WORKDIR /workspace