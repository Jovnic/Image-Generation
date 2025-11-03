FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir uv && uv sync

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "gan_api:app", "--host", "0.0.0.0", "--port", "8000"]
