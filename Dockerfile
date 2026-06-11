FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY weather_agent/ weather_agent/
COPY web/ web/
COPY config/ config/
COPY data/ data/
COPY memory/ memory/

# Create directories for runtime
RUN mkdir -p runs/cycles runs/audit runs/replay_bundles runs/objects runs/registry runs/truth_labels runs/truth_factory

# Set environment variables
ENV AGENT_PROFILE=dev
ENV AGENT_ENABLE_AUTH=0

EXPOSE 8000

CMD ["uvicorn", "weather_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
