FROM python:3.12-bookworm

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y gcc libffi-dev libssl-dev && apt-get install -y netcat-openbsd && apt-get clean

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir setuptools wheel build
RUN pip install --upgrade pip setuptools wheel

ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000" ]

