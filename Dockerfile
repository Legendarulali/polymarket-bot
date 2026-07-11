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

# După COPY, /app e deținut de root, deci userul non-root nu poate scrie
# history.json[.tmp] → [Errno 13] Permission denied. Dă-i botuser proprietatea.
RUN chown -R botuser:botuser /app

# Director de STARE PERSISTENTĂ: montează aici un volum (bind mount ./data:/data —
# vezi docker-compose.yml — sau un volum Railway pe /data). history.json (istoric
# analize + decizii AI + coada next_in_line) și logurile trăiesc pe volum, deci
# distrugerea/reconstruirea/repornirea containerului NU pierde nimic — la pornire
# botul le recitește direct de pe mașina gazdă. Fără volum montat, comportamentul
# rămâne cel vechi (stare efemeră în container). QUEUE_FILE e doar sursa de MIGRARE
# legacy a cozii (vechiul next_in_line.json separat).
ENV HISTORY_FILE=/data/history.json \
    QUEUE_FILE=/data/next_in_line.json \
    LOG_FILE=/data/logs/bot.log
RUN mkdir -p /data/logs && chown -R botuser:botuser /data

USER botuser

CMD ["python", "main.py"]
