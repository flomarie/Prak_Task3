FROM python:3.11-slim

COPY Prak_Task3/src/requirements.txt /root/Prak_Task3/src/requirements.txt

RUN chown -R root:root /root/Prak_Task3

WORKDIR /root/Prak_Task3/src
RUN pip3 install -r requirements.txt

COPY Prak_Task3/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP my_server

RUN chmod +x run.py
CMD ["python3", "run.py"]
