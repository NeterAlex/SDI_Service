FROM tiangolo/uvicorn-gunicorn:python3.11
# 配置信息
LABEL maintainer="NeterAlex <neteralex@outlook.com>"
WORKDIR /application
# 安装依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# 准备文件
COPY ./ /application
# 执行
CMD ["uvicorn", "main:app", "--proxy-headers" , "--host", "0.0.0.0", "--port", "5020"]