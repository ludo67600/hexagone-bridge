# ============================================================================
# Bridge IA PNJ — image Docker
# Compatible Hugging Face Spaces (port 7860), Render, Fly.io, Cloud Run...
# Le port est piloté par la variable d'environnement PORT.
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# Dépendances d'abord : le cache Docker évite de tout réinstaller à chaque push
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code du bridge
COPY main.py stt.py llm.py tts.py cache.py ./

# Le système de fichiers est éphémère : le cache va dans /tmp (toujours accessible
# en écriture, y compris quand l'hébergeur exécute le conteneur en non-root).
ENV CACHE_DB=/tmp/cache.sqlite3
ENV PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
