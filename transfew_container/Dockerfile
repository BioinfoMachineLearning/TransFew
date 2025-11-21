# 1) Download base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime


# 2. Avoid interactive tzdata issues etc.
ENV DEBIAN_FRONTEND=noninteractive


# 3. Install System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    vim \
 && rm -rf /var/lib/apt/lists/*


# 4. Set working directory
WORKDIR /workspace

# 5. Update conda environment & clean up
RUN conda install -n base -y \
    -c bioconda -c conda-forge -c pyg \
    diamond=0.9.14 \
    mmseqs2=13.45111 \
    pyg=2.3.1 && \
    conda clean --all

# 6. Copy pip requirements and install them
COPY requirements.txt .

# 7. Install pip packages
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy the rest of your project
COPY . .

# Fix line endings (Windows CRLF -> Unix LF)
RUN sed -i 's/\r$//' run_transfew.sh

# Make the wrapper script executable
RUN chmod +x run_transfew.sh

ENTRYPOINT ["./run_transfew.sh"] 
