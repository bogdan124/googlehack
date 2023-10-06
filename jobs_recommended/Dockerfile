# Use an official Python runtime as a parent image
FROM python:3.11.1-slim

# Set the working directory in the container
WORKDIR /jobs_recommended

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements2.txt .
RUN pip install -r requirements2.txt

# copy project
COPY . .