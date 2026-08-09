FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY chatbot_project/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY chatbot_project ./chatbot_project

EXPOSE 7860

ENTRYPOINT ["python", "-u", "-m", "chatbot_project.app.gradio_app"]
CMD ["--quick_demo"]
