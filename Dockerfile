FROM python:3.10.13-slim

WORKDIR /app

COPY requirements-prod.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-prod.txt

RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]