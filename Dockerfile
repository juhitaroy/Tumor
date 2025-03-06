# Use the official Python image
FROM python:3.10

# Set the working directory to the Alzheimers folder
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire Alzheimers folder into the Docker image
COPY . .

# Expose port 8000
EXPOSE 8001

# Set the command to run the FastAPI app
CMD ["uvicorn", "server_tumor:app", "--host", "0.0.0.0", "--port", "8001"]

