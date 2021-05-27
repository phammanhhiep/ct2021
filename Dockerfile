FROM continuumio/miniconda3

LABEL "maintainer"="hieppm.work@gmail.com"

ENV MYUSER ct2021

# Create non-root user and set permission to conda directory
RUN useradd -m $MYUSER && \
    chown -R $MYUSER /opt/conda 
USER $MYUSER
WORKDIR /home/$MYUSER

# Initialize conda
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Copy applications files and create mount directory
COPY src ./src
COPY requirements.txt ./
RUN mkdir production

# Switch shell sh (default in Linux) to bash
SHELL ["/bin/bash", "-c"]

# Give bash access to Anaconda
RUN echo "source activate base" >> ~/.bashrc && \
    echo "PYTHONPATH=/home/$MYUSER" >> ~/.bashrc && \
    source /home/$MYUSER/.bashrc && \
    export CONDA_ALWAYS_YES="true" && \
    conda config --append channels conda-forge && \
    conda config --append channels pytorch && \
    conda install --file requirements.txt && \
    unset CONDA_ALWAYS_YES