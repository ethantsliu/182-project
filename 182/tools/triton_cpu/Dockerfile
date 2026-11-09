# CPU image for running the repo's Triton kernels under the Triton interpreter (TRITON_INTERPRET=1), and for
# compiling them for sm_90a without a GPU (compile_check.py; the triton wheel bundles ptxas).
# Default: torch 2.10.0 (the GPU env's pin, requirements.txt) and triton 3.6.0, the version its wheels require
# (cu128 and cpu alike), so kernels go through the same frontend as on the node. numpy < 2.3: the interpreter
# calls int() on 1-element arrays. Other pairs: run.sh TRITON_CPU_VERSION=3.5.1 (torch 2.9.1).
FROM python:3.12-slim
ARG TORCH=2.10.0
ARG TRITON=3.6.0
RUN pip install --no-cache-dir torch==${TORCH} --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir triton==${TRITON} numpy==2.2.6 pytest==8.4.2
ENV TRITON_INTERPRET=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /repo
