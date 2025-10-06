# last updated Mar 25 2025, 11:00am
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# ---------- 1. 安装 Java (OpenJDK 11) ----------
RUN apt-get update && apt-get install -y openjdk-11-jdk curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- 2. 设置 JAVA_HOME ----------
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# ---------- 3. 安装 Python 依赖 ----------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 4. 暴露 JupyterLab 端口 ----------
EXPOSE 8888

# ---------- 5. 创建挂载点 ----------
VOLUME /app

# ---------- 6. 启用 JupyterLab ----------
ENV JUPYTER_ENABLE_LAB=yes

# ---------- 7. 默认启动命令 ----------
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]

