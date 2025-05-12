# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
# (Not strictly needed for this app, but useful if you add a web interface)
EXPOSE 80

# Define environment variable
ENV DISPLAY=:0

# Install OpenCV dependencies
RUN apt-get update && \
    apt-get install -y libgl1 libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Run main.py when the container launches
CMD ["python", "main.py"]