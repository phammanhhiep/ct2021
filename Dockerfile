FROM python:3.7.10-slim

LABEL "maintainer"="hieppm.work@gmail.com"
USER hieppm

RUN mkdir -p ./projects/ct2021
WORKDIR ./projeccts/ct2021
COPY ./* ./

RUN conda create -n prod -f requirements.txt

CMD ["./eval.sh"]