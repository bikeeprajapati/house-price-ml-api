FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirement file first (enables Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Start API using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
