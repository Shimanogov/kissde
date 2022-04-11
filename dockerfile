FROM pytorch/pytorch:latest
ADD requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt