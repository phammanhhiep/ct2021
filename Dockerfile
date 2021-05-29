# NOTICE: user environments and conda init is add to .bashrc for convenience in
# case the coresponding containers are run with interactive shells.

FROM continuumio/miniconda3

LABEL "maintainer"="hieppm.work@gmail.com"

ARG requirements
ARG user

# Switch shell sh (default in Linux) to bash
SHELL ["/bin/bash", "-c"]

ENV MYUSER $user
ENV MYREQ $requirements

# Create non-root user and set permission to conda directory
RUN useradd -m $MYUSER && chown -R $MYUSER /opt/conda 
USER $MYUSER
WORKDIR /home/$MYUSER

# Copy applications files and create mount directory
COPY src ./src
COPY $requirements ./
COPY README.md ./
RUN mkdir artifacts

# Give bash access to Anaconda and install packages
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo "export PYTHONPATH=/home/$MYUSER" >> ~/.bashrc && \
    source /home/$MYUSER/.bashrc && \
    export CONDA_ALWAYS_YES="true" && \
    conda config --append channels conda-forge && \
    conda config --append channels pytorch && \
    conda install --file $MYREQ && \
    unset CONDA_ALWAYS_YES