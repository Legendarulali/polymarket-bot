# Etapa de build: compilează dependențele care au extensii native,
# fără a păstra toolchain-ul (build-essential) în imaginea finală.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Etapa finală: imagine minimă, fără compilator, rulată ca user non-root.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 botuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

USER botuser

CMD ["python", "main.py"]
