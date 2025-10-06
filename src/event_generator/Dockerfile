FROM python:3.11-slim
WORKDIR /app
COPY event_generator.py .
RUN pip install kafka-python
CMD ["python", "event_generator.py"]
