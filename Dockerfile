FROM docker.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

LABEL maintainer="FA-Kara WebUI"
LABEL description="Karaoke lyrics alignment tool with WebUI"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 安装系统依赖 + Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 PyTorch Nightly (支持 RTX 50 系列 Blackwell sm_120)
RUN pip install --no-cache-dir --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安装其他 Python 依赖
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# 预下载 NLTK 数据
RUN python -c "import nltk; nltk.download('cmudict', download_dir='/usr/share/nltk_data')"

# 复制应用代码
COPY FA-Kara/ ./FA-Kara/
COPY app.py .

# 创建非 root 用户 (如果 UID 1000 已存在则跳过)
RUN id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 appuser; \
    chown -R $(id -nu 1000):$(id -ng 1000) /app
USER 1000

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "app.py"]

