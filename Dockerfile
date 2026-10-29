FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY mlforge_core ./mlforge_core
COPY backend ./backend
COPY frontend ./frontend
COPY sample_data ./sample_data

RUN pip install --no-cache-dir .
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
