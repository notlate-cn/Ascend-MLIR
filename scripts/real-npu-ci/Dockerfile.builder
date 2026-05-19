# syntax=docker/dockerfile:1

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      bash \
      ca-certificates \
      ccache \
      clang \
      cmake \
      file \
      g++ \
      gcc \
      git \
      lld \
      make \
      ninja-build \
      openssh-client \
      patchelf \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      rsync \
      sudo \
      tar \
      time \
      tzdata \
      unzip \
      wget \
      xz-utils \
      zlib1g-dev && \
    python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir numpy pybind11 nanobind PyYAML pygments && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_LLVM=0
ARG LLVM_REPO_URL=https://github.com/llvm/llvm-project.git
ARG LLVM_REF=2078da43e25a4623cab2d0d60decddf709aaea28
ARG LLVM_BUILD_TYPE=Release
ARG LLVM_ENABLE_ASSERTIONS=ON
ARG LLVM_JOBS=4

ENV ASCEND_HOME_PATH=/data/nyh/Ascend/latest
ENV ASCEND_TOOLKIT_HOME=/data/nyh/Ascend/latest
ENV LLVM_BUILD_DIR=/opt/llvm/build
ENV CC=clang
ENV CXX=clang++

RUN mkdir -p /workspace /jobs /opt/ascend-mlir-ci /opt/llvm

RUN if [ "${BUILD_LLVM}" = "1" ]; then \
      mkdir -p /opt/llvm/src /opt/llvm/build && \
      git init /opt/llvm/src && \
      cd /opt/llvm/src && \
      git remote add origin "${LLVM_REPO_URL}" && \
      git fetch --depth 1 origin "${LLVM_REF}" && \
      git checkout FETCH_HEAD && \
      cd /opt/llvm/build && \
      cmake -G Ninja /opt/llvm/src/llvm \
        -DCMAKE_BUILD_TYPE="${LLVM_BUILD_TYPE}" \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD=host \
        -DLLVM_ENABLE_ASSERTIONS="${LLVM_ENABLE_ASSERTIONS}" \
        -DLLVM_ENABLE_RTTI=ON \
        -DLLVM_BUILD_EXAMPLES=OFF \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
        -DMLIR_PYTHON_BINDINGS_LIBRARY=nanobind \
        -DPython3_EXECUTABLE="$(command -v python3)" \
        -DLLVM_ENABLE_LIBEDIT=OFF && \
      cmake --build . --target all -j"${LLVM_JOBS}"; \
    fi

COPY . /opt/ascend-mlir-ci/

RUN if [ -d /opt/ascend-mlir-ci/llvm-build/lib/cmake/mlir ]; then \
      if [ -d /opt/llvm/build/lib/cmake/mlir ]; then \
        echo "image already contains /opt/llvm/build; do not combine BUILD_LLVM=1 and embedded llvm-build context" >&2; \
        exit 1; \
      fi; \
      mkdir -p /opt/llvm/build && \
      cp -a /opt/ascend-mlir-ci/llvm-build/. /opt/llvm/build/ && \
      rm -rf /opt/ascend-mlir-ci/llvm-build; \
    fi && \
    chmod +x /opt/ascend-mlir-ci/collect-plog.sh \
             /opt/ascend-mlir-ci/run-real-npu-job.sh \
             /opt/ascend-mlir-ci/submit-910c.sh \
             /opt/ascend-mlir-ci/docker-run-910c.sh \
             /opt/ascend-mlir-ci/build-aarch64-image.sh

WORKDIR /workspace

ENTRYPOINT ["/opt/ascend-mlir-ci/run-real-npu-job.sh"]
