# ChromaDB Server Installation & Configuration Guide - Ubuntu 22.04

This guide provides comprehensive instructions for installing and configuring ChromaDB server locally on Ubuntu 22.04 LTS.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Prerequisites Installation](#prerequisites-installation)
3. [ChromaDB Installation Methods](#chromadb-installation-methods)
4. [Server Configuration](#server-configuration)
5. [Client Setup & Testing](#client-setup--testing)
6. [Security Configuration](#security-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 5GB free space minimum
- **SQLite**: Version 3.35 or higher

### Recommended Development Setup
- **RAM**: 16GB for optimal performance
- **CPU**: 4+ cores
- **Storage**: SSD with 20GB+ free space
- **Network**: Stable internet connection

## Prerequisites Installation

### 1. Update System Packages

```bash
# Update package lists and upgrade system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential curl wget git vim
```

### 2. Install Python 3.11+

```bash
# Add deadsnakes PPA for latest Python versions
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Install Python 3.11 and related packages
sudo apt install -y python3.11 python3.11-pip python3.11-venv python3.11-dev

# Set Python 3.11 as default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip3.11 1

# Verify installation
python3 --version
pip3 --version
```

### 3. Check SQLite Version

```bash
# Check SQLite version (must be 3.35+)
sqlite3 --version

# If version is older than 3.35, update it
sudo apt install -y sqlite3 libsqlite3-dev
```

## ChromaDB Installation Methods

### Method 1: Docker Installation (Recommended for Production)

#### Install Docker

```bash
# Remove old Docker versions
sudo apt remove -y docker docker-engine docker.io containerd runc

# Install Docker dependencies
sudo apt install -y ca-certificates gnupg lsb-release

# Add Docker GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker --version
docker compose version
```

#### Run ChromaDB Server with Docker

```bash
# Create data directory
mkdir -p ~/chroma-data
chmod 755 ~/chroma-data

# Run ChromaDB server container
docker run -d \
  --name chromadb-server \
  --restart unless-stopped \
  -p 8000:8000 \
  -v ~/chroma-data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e PERSIST_DIRECTORY=/chroma/chroma \
  -e ANONYMIZED_TELEMETRY=TRUE \
  chromadb/chroma:latest

# Check container status
docker ps
docker logs chromadb-server
```

#### Docker Compose Setup (Alternative)

```bash
# Create docker-compose.yml
cat > ~/chroma-docker-compose.yml << 'EOF'
version: '3.9'

networks:
  chroma-net:
    driver: bridge

services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb-server
    restart: unless-stopped
    volumes:
      - ./chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
      - ANONYMIZED_TELEMETRY=TRUE
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
    ports:
      - "8000:8000"
    networks:
      - chroma-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

# Run with Docker Compose
cd ~
docker compose -f chroma-docker-compose.yml up -d

# Check logs
docker compose -f chroma-docker-compose.yml logs -f
```

### Method 2: Native Python Installation

#### Create Python Virtual Environment

```bash
# Create project directory
mkdir -p ~/chromadb-server
cd ~/chromadb-server

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Install ChromaDB

```bash
# Install ChromaDB server package
pip install chromadb

# For client-only installation (if needed separately)
# pip install chromadb-client

# Verify installation
python -c "import chromadb; print(f'ChromaDB version: {chromadb.__version__}')"
```

#### Run ChromaDB Server

```bash
# Create data directory
mkdir -p ~/chromadb-data

# Run ChromaDB server
chroma run --host 0.0.0.0 --port 8000 --path ~/chromadb-data

# Run in background with nohup
nohup chroma run --host 0.0.0.0 --port 8000 --path ~/chromadb-data > chromadb.log 2>&1 &

# Check if server is running
ps aux | grep chroma
curl http://localhost:8000/api/v1/heartbeat
```

#### Create Systemd Service (Optional)

```bash
# Create systemd service file
sudo tee /etc/systemd/system/chromadb.service << EOF
[Unit]
Description=ChromaDB Server
After=network.target
Wants=network.target

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$HOME/chromadb-server
Environment=PATH=$HOME/chromadb-server/venv/bin
ExecStart=$HOME/chromadb-server/venv/bin/chroma run --host 0.0.0.0 --port 8000 --path $HOME/chromadb-data
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
sudo systemctl daemon-reload
sudo systemctl enable chromadb
sudo systemctl start chromadb

# Check service status
sudo systemctl status chromadb
```

## Server Configuration

### Environment Variables

```bash
# Create environment configuration file
cat > ~/chromadb-server/.env << 'EOF'
# Server Configuration
CHROMA_SERVER_HOST=0.0.0.0
CHROMA_SERVER_HTTP_PORT=8000
CHROMA_SERVER_CORS_ALLOW_ORIGINS=["*"]

# Persistence Configuration
IS_PERSISTENT=TRUE
PERSIST_DIRECTORY=/path/to/chroma/data
CHROMA_DB_IMPL=duckdb+parquet

# Performance Settings
CHROMA_SERVER_MAX_REQUEST_SIZE=33554432  # 32MB
CHROMA_SERVER_REQUEST_TIMEOUT=60

# Logging Configuration
CHROMA_LOG_LEVEL=INFO
ANONYMIZED_TELEMETRY=TRUE

# Authentication (if enabled)
# CHROMA_SERVER_AUTHN_CREDENTIALS_FILE=/path/to/credentials
# CHROMA_SERVER_AUTHN_PROVIDER=basic
EOF
```

### Custom Configuration File

```bash
# Create chromadb config file
mkdir -p ~/.config/chroma
cat > ~/.config/chroma/config.yaml << 'EOF'
# ChromaDB Configuration
chroma_server_host: "0.0.0.0"
chroma_server_http_port: 8000
chroma_db_impl: "duckdb+parquet"
persist_directory: "/home/$USER/chromadb-data"
is_persistent: true

# CORS settings
chroma_server_cors_allow_origins: ["*"]

# Performance settings
max_request_size: 33554432  # 32MB
request_timeout: 60

# Logging
log_level: "INFO"
anonymized_telemetry: true
EOF
```

## Client Setup & Testing

### Install ChromaDB Client

```bash
# Install client in a separate environment
python3 -m venv ~/chromadb-client-env
source ~/chromadb-client-env/bin/activate
pip install chromadb-client

# Or install full chromadb package
# pip install chromadb
```

### Basic Client Testing

```python
# Create test script: test_chromadb.py
cat > ~/test_chromadb.py << 'EOF'
#!/usr/bin/env python3
import chromadb
import time

def test_chromadb_connection():
    try:
        # Connect to ChromaDB server
        client = chromadb.HttpClient(host="localhost", port=8000)
        
        # Test heartbeat
        heartbeat = client.heartbeat()
        print(f"✅ Server heartbeat: {heartbeat}")
        
        # List existing collections
        collections = client.list_collections()
        print(f"📚 Existing collections: {[c.name for c in collections]}")
        
        # Create test collection
        collection_name = f"test_collection_{int(time.time())}"
        collection = client.create_collection(name=collection_name)
        print(f"📝 Created collection: {collection_name}")
        
        # Add test documents
        test_docs = [
            "This is a test document about Python programming.",
            "ChromaDB is a vector database for AI applications.",
            "Ubuntu 22.04 is a stable Linux distribution."
        ]
        
        collection.add(
            documents=test_docs,
            metadatas=[
                {"topic": "programming", "language": "python"},
                {"topic": "database", "type": "vector"},
                {"topic": "operating_system", "type": "linux"}
            ],
            ids=[f"doc_{i}" for i in range(len(test_docs))]
        )
        print(f"📄 Added {len(test_docs)} documents")
        
        # Query test
        results = collection.query(
            query_texts=["database"],
            n_results=2
        )
        print(f"🔍 Query results: {len(results['documents'][0])} documents found")
        
        # Cleanup
        client.delete_collection(name=collection_name)
        print(f"🗑️  Cleaned up test collection")
        
        print("\n✅ All tests passed! ChromaDB is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error testing ChromaDB: {e}")
        return False

if __name__ == "__main__":
    test_chromadb_connection()
EOF

# Run test
python3 ~/test_chromadb.py
```

### Advanced Client Configuration

```python
# Create advanced client example: advanced_client.py
cat > ~/advanced_client.py << 'EOF'
#!/usr/bin/env python3
import chromadb
from chromadb.config import Settings

def create_advanced_client():
    """Create ChromaDB client with advanced configuration"""
    
    # Configure client settings
    settings = Settings(
        chroma_server_host="localhost",
        chroma_server_http_port=8000,
        chroma_server_ssl_enabled=False,
        chroma_server_cors_allow_origins=["*"],
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_grpc_port=None,
    )
    
    # Create HTTP client
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False,
        headers={}  # Add authentication headers if needed
    )
    
    return client

def create_collection_with_embedding_function():
    """Create collection with custom embedding function"""
    from chromadb.utils import embedding_functions
    
    client = create_advanced_client()
    
    # Use OpenAI embeddings (requires API key)
    # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    #     api_key="your-openai-api-key",
    #     model_name="text-embedding-ada-002"
    # )
    
    # Use SentenceTransformer embeddings (default)
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name="advanced_collection",
        embedding_function=default_ef,
        metadata={"description": "Collection with custom embedding function"}
    )
    
    return collection

if __name__ == "__main__":
    client = create_advanced_client()
    print("✅ Advanced ChromaDB client created successfully")
    
    collection = create_collection_with_embedding_function()
    print("✅ Collection with embedding function created")
EOF
```

## Security Configuration

### Basic Authentication Setup

```bash
# Create authentication credentials file
mkdir -p ~/chromadb-auth
cat > ~/chromadb-auth/credentials.txt << 'EOF'
# Format: username:password (plain text for basic auth)
admin:your_secure_password_here
user1:another_secure_password
EOF

# Secure the credentials file
chmod 600 ~/chromadb-auth/credentials.txt
```

### SSL/TLS Configuration (Optional)

```bash
# Generate self-signed certificate for development
mkdir -p ~/chromadb-ssl
cd ~/chromadb-ssl

# Generate private key
openssl genrsa -out chromadb.key 2048

# Generate certificate signing request
openssl req -new -key chromadb.key -out chromadb.csr -subj "/CN=localhost"

# Generate self-signed certificate
openssl x509 -req -in chromadb.csr -signkey chromadb.key -out chromadb.crt -days 365

# Set proper permissions
chmod 600 chromadb.key
chmod 644 chromadb.crt
```

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 8000/tcp # ChromaDB

# Check firewall status
sudo ufw status verbose
```

## Performance Tuning

### System Optimization

```bash
# Increase file descriptor limits
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize TCP settings for database connections
sudo tee -a /etc/sysctl.conf << 'EOF'
# ChromaDB optimizations
net.core.somaxconn = 1024
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_keepalive_time = 300
EOF

# Apply settings
sudo sysctl -p
```

### ChromaDB Performance Settings

```bash
# Create performance-optimized configuration
cat > ~/chromadb-performance.env << 'EOF'
# Performance Configuration
CHROMA_SERVER_MAX_REQUEST_SIZE=134217728  # 128MB
CHROMA_SERVER_REQUEST_TIMEOUT=120
CHROMA_SERVER_MAX_WORKERS=4
CHROMA_SERVER_WORKER_CLASS=uvicorn.workers.UvicornWorker

# Memory settings
CHROMA_MEMORY_LIMIT=8GB
CHROMA_CACHE_SIZE=2GB

# Batch processing
CHROMA_BATCH_SIZE=1000
CHROMA_MAX_BATCH_SIZE=10000

# DuckDB specific optimizations
DUCKDB_MEMORY_LIMIT=4GB
DUCKDB_THREADS=4
EOF
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: SQLite Version Too Old

```bash
# Check current SQLite version
sqlite3 --version

# If version < 3.35, compile from source
wget https://www.sqlite.org/2023/sqlite-autoconf-3420000.tar.gz
tar xzf sqlite-autoconf-3420000.tar.gz
cd sqlite-autoconf-3420000
./configure --prefix=/usr/local
make
sudo make install

# Update library path
echo '/usr/local/lib' | sudo tee /etc/ld.so.conf.d/sqlite3.conf
sudo ldconfig
```

#### Issue 2: Permission Denied Errors

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ~/chromadb-data
chmod -R 755 ~/chromadb-data

# Fix systemd service permissions
sudo systemctl edit chromadb
# Add:
# [Service]
# User=your_username
# Group=your_group
```

#### Issue 3: Port Already in Use

```bash
# Check what's using port 8000
sudo netstat -tulpn | grep :8000
sudo lsof -i :8000

# Kill process using port
sudo kill -9 $(sudo lsof -t -i:8000)

# Or use different port
chroma run --host 0.0.0.0 --port 8001 --path ~/chromadb-data
```

#### Issue 4: Memory Issues

```bash
# Check memory usage
free -h
top -p $(pgrep -f chroma)

# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Debugging Commands

```bash
# Check ChromaDB logs
# For Docker:
docker logs chromadb-server

# For systemd service:
sudo journalctl -u chromadb -f

# For manual installation:
tail -f ~/chromadb-server/chromadb.log

# Test network connectivity
curl -v http://localhost:8000/api/v1/heartbeat
curl -v http://localhost:8000/api/v1/version

# Check process status
ps aux | grep chroma
netstat -tulpn | grep 8000
```

### Log Analysis

```bash
# Enable debug logging
export CHROMA_LOG_LEVEL=DEBUG

# Create log analysis script
cat > ~/analyze_chroma_logs.py << 'EOF'
#!/usr/bin/env python3
import re
from collections import Counter

def analyze_logs(log_file):
    with open(log_file, 'r') as f:
        logs = f.readlines()
    
    # Count log levels
    levels = Counter()
    errors = []
    
    for line in logs:
        if 'ERROR' in line:
            levels['ERROR'] += 1
            errors.append(line.strip())
        elif 'WARNING' in line:
            levels['WARNING'] += 1
        elif 'INFO' in line:
            levels['INFO'] += 1
        elif 'DEBUG' in line:
            levels['DEBUG'] += 1
    
    print("Log Level Summary:")
    for level, count in levels.items():
        print(f"  {level}: {count}")
    
    if errors:
        print("\nRecent Errors:")
        for error in errors[-5:]:  # Last 5 errors
            print(f"  {error}")

if __name__ == "__main__":
    analyze_logs("chromadb.log")
EOF
```

This comprehensive guide provides everything needed to install, configure, and troubleshoot ChromaDB server on Ubuntu 22.04. Choose the installation method that best fits your needs - Docker for production deployments or native installation for development environments.