# NOTICE: user environments and conda init is add to .bashrc for convenience in
# case the coresponding containers are run with interactive shells.

FROM continuumio/miniconda3

LABEL "maintainer"="hieppm.work@gmail.com"

ARG requirements
ARG user
ARG project_dir=ct2021

# Switch shell sh (default in Linux) to bash
SHELL ["/bin/bash", "-c"]

ENV MYUSER $user
ENV MYREQ $requirements
ENV PROJECT_DIR $project_dir

# Create non-root user and set permission to conda directory
RUN useradd -m $MYUSER && \
    chown -R $MYUSER /opt/conda 
USER $MYUSER
WORKDIR /home/$MYUSER

# Copy applications files and create mount directory
RUN mkdir -p $PROJECT_DIR/artifacts \
    $PROJECT_DIR/tools
COPY src $PROJECT_DIR/src
COPY scripts $PROJECT_DIR/scripts
COPY Dockerfile $MYREQ README.md $PROJECT_DIR/

# Give bash access to Anaconda and install packages
RUN export CONDA_ALWAYS_YES="true" && \
    conda config --append channels conda-forge && \
    conda config --append channels pytorch && \
    conda install --file $PROJECT_DIR/$MYREQ && \
    unset CONDA_ALWAYS_YES