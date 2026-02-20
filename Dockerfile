FROM python:3.10.13-slim

WORKDIR /app

COPY requirements-prod.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements-prod.txt

# Install spaCy model directly
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

COPY . .

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]