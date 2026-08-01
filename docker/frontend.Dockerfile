# Packaging only (Docker Milestone) -- does not modify anything under
# production/frontend/. Builds the Streamlit dashboard (dashboard.py) as a
# standalone container.
#
# Installs only the packages dashboard.py itself imports (streamlit,
# requests, pandas, websocket-client) rather than the full
# requirements.txt -- that file also pulls in torch/ultralytics/
# torch-geometric etc. for the backend/CV track, none of which
# dashboard.py touches. Deliberately not editing requirements.txt itself;
# this is a packaging-layer choice about what THIS image installs.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit pandas requests websocket-client

COPY production/frontend/ production/frontend/

EXPOSE 8501

CMD ["streamlit", "run", "production/frontend/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
