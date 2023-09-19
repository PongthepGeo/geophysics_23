FROM python:3.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential curl vim git && \
    apt-get clean

# Install Miniconda3 and set PATH
RUN curl -Lo miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    echo "export PATH=/opt/conda/bin:${PATH}" >> ~/.bashrc 

ENV PATH="/opt/conda/bin:${PATH}"

# Ensure Python 3.7 is installed in Miniconda
RUN conda install -y python=3.7

# Install pgcore using conda and cleanup
RUN conda install -c gimli -c conda-forge pgcore && \
    conda clean -a -y

# Set up the working directory
WORKDIR /workspace

CMD ["/bin/bash"]

