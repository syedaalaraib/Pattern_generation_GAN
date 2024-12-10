# Use the official Python base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install the necessary dependencies (like PyTorch, Flask, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app will run on
EXPOSE 5000

# Command to run your Python application (e.g., Flask app or script)
CMD ["python", "textile_gan.py"]
