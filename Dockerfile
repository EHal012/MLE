FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless tini procps && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir jupyterlab

EXPOSE 8888
CMD ["tini","--","jupyter","lab","--ip=0.0.0.0","--port=8888","--no-browser","--allow-root","--ServerApp.token="]
