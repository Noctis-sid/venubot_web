FROM python:3.11-slim

WORKDIR /app
COPY . /app

# Install system build deps (temporary) and clean up to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Use the same port Render injects
EXPOSE 8080

# Example start command — change if your main file/name is different
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"]
