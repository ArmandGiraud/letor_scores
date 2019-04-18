FROM python:3.7.2-stretch

COPY . .
EXPOSE 4545
RUN pip install -r requirements.txt