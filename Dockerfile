FROM python:3.13-slim

LABEL maintainer="Risk Data Team"
LABEL description="Credit Risk Scoring — API + Dashboard"
LABEL version="1.0.0"

WORKDIR /app

# Dépendances système minimales pour numba/llvmlite (SHAP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python (layer mis en cache si requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Code source uniquement (data/ et models/ sont dans des volumes Docker)
COPY src/ ./src/
COPY dashboard/ ./dashboard/

# Créer les répertoires attendus par le code
RUN mkdir -p data/raw data/processed data/sample models docs

# Exposer les ports API et Dashboard
EXPOSE 8000 8501

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
