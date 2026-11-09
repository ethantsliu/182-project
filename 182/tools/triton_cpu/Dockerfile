# CPU image for running the repo's Triton kernels under the Triton interpreter (TRITON_INTERPRET=1).
# triton 3.6.0 is the version torch 2.10 (the GPU env's pin, requirements.txt) ships with, so the kernels go
# through the same frontend as on the node. numpy < 2.3: the interpreter calls int() on 1-element arrays.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir triton==3.6.0 numpy==2.2.6 pytest==8.4.2
ENV TRITON_INTERPRET=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /repo
