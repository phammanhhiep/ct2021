FROM continuumio/miniconda3

LABEL "maintainer"="hieppm.work@gmail.com"

ENV MYUSER ct2021

RUN useradd -m $MYUSER
USER $MYUSER
WORKDIR /home/$MYUSER

# Copy applications files
COPY src ./src
COPY eval.sh ./
COPY train.sh ./
COPY requirements.txt ./

# Switch shell sh (default in Linux) to bash
SHELL ["/bin/bash", "-c"]

# Give bash access to Anaconda
RUN echo "source activate env" >> ~/.bashrc && \
    source /home/$MYUSER/.bashrc && \
    export CONDA_ALWAYS_YES="true" && \
    conda config --append channels conda-forge && \
    conda config --append channels pytorch && \
    conda install --file requirements.txt && \
    unset CONDA_ALWAYS_YES \