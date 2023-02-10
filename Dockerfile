FROM jupyter/scipy-notebook


RUN pip install joblib

COPY train.py ./train.py
COPY test.py ./test.py

RUN python3 train.py

