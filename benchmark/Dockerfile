FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn numpy
COPY wire_server.py .
EXPOSE 9000
CMD ["uvicorn", "wire_server:app", "--host", "0.0.0.0", "--port", "9000"]
