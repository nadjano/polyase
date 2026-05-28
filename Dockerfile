# ============================================================
# Docker image for PolyASE + JupyterLab
#
# Source: https://pypi.org/project/polyase/
# Requires Python >=3.11, <3.13  →  we use 3.12
#
# pyranges is installed via conda (bioconda) to avoid the
# known pip build failures on some platforms (llvmlite/numba).
#
# Build:
#   docker build -t polyase:latest .
#
# Run JupyterLab (mount your notebooks folder):
#   docker run --rm -p 8888:8888 \
#     -v $(pwd)/notebooks:/home/user/notebooks \
#     polyase:latest
#
#   Then open http://localhost:8888 in your browser.
#
# Run a script directly:
#   docker run --rm -v $(pwd):/home/user/work polyase:latest \
#     python /home/user/work/my_script.py
# ============================================================

# Miniforge gives us conda + mamba on a slim Debian base,
# matching the "conda create ... python=3.12" workflow exactly.
FROM condaforge/miniforge3:latest

LABEL maintainer="nadja.franziska.nolte@nib.si"
LABEL description="PolyASE – allele-specific expression in polyploids"
LABEL org.opencontainers.image.source="https://github.com/NIB-SI/polyase"

# ── non-root user ────────────────────────────────────────────
RUN useradd --create-home --shell /bin/bash user
WORKDIR /home/user

# ── conda environment ────────────────────────────────────────
# Mirrors the official install command exactly:
#   conda create -n polyase python=3.12 ipykernel pip
#   pip install polyase
#   pip install pyranges  (via conda here to avoid build issues)
#
# We also add:
#   jupyterlab   – so notebooks can be opened in the browser
#   ipywidgets   – interactive widgets used by some PolyASE plots
#   numba        – installed via conda-forge to avoid llvmlite issues
RUN mamba create -y -n polyase python=3.12 ipykernel pip \
    && mamba install -y -n polyase \
        -c conda-forge \
        -c bioconda \
        jupyterlab \
        ipywidgets \
        numba \
        pyranges \
    && conda run -n polyase pip install --no-cache-dir polyase \
    && conda clean -afy

# ── make the environment's jupyter the default command ───────
ENV PATH="/opt/conda/envs/polyase/bin:$PATH"

# ── register the kernel so notebooks can pick it up ─────────
RUN conda run -n polyase python -m ipykernel install \
        --user --name polyase --display-name "Python (polyase)"

# ── directories for user data ────────────────────────────────
RUN mkdir -p /home/user/notebooks /home/user/data \
    && chown -R user:user /home/user

VOLUME ["/home/user/notebooks", "/home/user/data"]

USER user

# ── start JupyterLab on port 8888 ───────────────────────────
# Token/password auth is disabled for convenience on localhost.
# For shared or remote servers, remove the token/password lines
# and let Jupyter generate a token instead.
EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''", \
     "--notebook-dir=/home/user/notebooks"]
