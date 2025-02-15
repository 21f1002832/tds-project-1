FROM python:3.12-slim-bookworm

# Install system dependencies (curl and ca-certificates)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm

# Install Prettier globally (ESSENTIAL)
RUN npm install -g prettier@3.4.2

# Download and run uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

RUN mkdir -p /data

COPY . . 

CMD ["uv", "run", "app.py"]