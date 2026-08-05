# LOGOS RunPod training image (single-node, 1x-8x H100 SXM).
# Build: docker build -t canivel/logos:latest -f docker/Dockerfile .
# The spot-hardening contract (PLAN.md s5): checkpoints every 30 min + on
# SIGTERM to the network volume mounted at /workspace; auto-resume on restart.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN pip install --no-cache-dir uv
WORKDIR /opt/logos
COPY pyproject.toml README.md PLAN.md ./
COPY src/ src/
COPY validation/ validation/
COPY manifests/ manifests/
RUN uv pip install --system -e ".[tracking,export,evalsuite]" --no-deps \
    && uv pip install --system wandb gguf "lm-eval==0.4.9" pyyaml datasets transformers safetensors scipy pandas matplotlib

# Runs resume automatically: the entrypoint re-invokes the same manifest run;
# the trainer finds ckpt_latest.pt on the volume and continues.
ENV LOGOS_RUNS_DIR=/workspace/runs \
    LOGOS_DATA_DIR=/workspace/data
ENTRYPOINT ["python", "-m", "logos.cli"]
