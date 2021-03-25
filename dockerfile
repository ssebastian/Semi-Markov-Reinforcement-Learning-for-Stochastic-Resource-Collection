FROM python:3.8.8-buster

RUN apt-get update
RUN apt-get install libspatialindex-dev libgdal-dev ffmpeg -y

WORKDIR /mnt
COPY create_db.py /mnt
COPY requirements.txt /mnt
COPY db_schema /mnt

RUN pip install -r requirements.txt
RUN python create_db.py
CMD python -O main.py