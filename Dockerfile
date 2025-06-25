# Use a specific patch version for reproducibility
FROM python:3.13.3-slim AS base

# Avoid running as root when installing OS packages
USER root

# Metadata helps later maintainers (and you in six months) know who owns this image
LABEL maintainer="Eduardo <psnedujime@gmail.com>" \
      description="ML TensorFlow like module."

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Install OS deps + git + bash, then clean up in one layer
USER root

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends git bash \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /etc/sudoers.d \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN git config --global --add safe.directory /app
RUN chsh -s /bin/bash $USERNAME

USER $USERNAME

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=$USERNAME:$USERNAME . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import socket; \
                 s=socket.socket(); \
                 s.bind(('127.0.0.1', 0)); \
                 s.close()" || exit 1

# Use an ENTRYPOINT if you always run the same command
CMD [ "bash" ]