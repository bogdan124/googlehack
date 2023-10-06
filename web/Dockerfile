# Use an official Python runtime as a parent image
FROM python:3.11.1-slim

# Set the working directory in the container
WORKDIR /web

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
COPY requirements.txt .t .
RUN pip install -r requirements.txt

# copy project
COPY . .