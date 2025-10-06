# last updated Oct 2025
# ✅ 使用自带 OpenJDK 17 的官方镜像，避免 apt 找不到包的问题
FROM openjdk:17-slim

# 设置非交互模式，避免 apt 卡住
ENV DEBIAN_FRONTEND=noninteractive

# 安装 Python、pip、procps（支持 ps 命令）和 bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip procps bash && \
    rm -rf /var/lib/apt/lists/*

# 创建 python / pip 命令软链接（某些镜像只带 python3 命令）
RUN ln -s /usr/bin/python3 /usr/bin/python || true && \
    ln -s /usr/bin/pip3 /usr/bin/pip || true


# 设置工作目录
WORKDIR /app

# 复制依赖文件到容器
COPY requirements.txt ./

# 安装 Python 依赖（建议 requirements.txt 里包括 pyspark、jupyterlab 等）
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 JupyterLab 默认端口
EXPOSE 8888

# 挂载 notebooks 的工作目录
VOLUME /app

# 启用 JupyterLab
ENV JUPYTER_ENABLE_LAB=yes

# 启动命令：运行 JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
