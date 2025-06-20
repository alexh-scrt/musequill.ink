# MuseQuill.Ink Development Environment Setup Instructions

This document provides comprehensive instructions for setting up the complete development environment for MuseQuill.Ink, including all standalone hosted technologies.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Core Dependencies Installation](#core-dependencies-installation)
3. [Database Setup](#database-setup)
4. [Vector Storage Setup](#vector-storage-setup)
5. [Knowledge Graph Setup (Neo4j)](#knowledge-graph-setup-neo4j)
6. [Caching Setup (Redis)](#caching-setup-redis)
7. [Search APIs Configuration](#search-apis-configuration)
8. [Application Configuration](#application-configuration)
9. [Development Tools Setup](#development-tools-setup)
10. [Verification and Testing](#verification-and-testing)
11. [Docker Setup (Alternative)](#docker-setup-alternative)
12. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+, macOS 10.15+, Windows 10+ (with WSL2 recommended)
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space minimum
- **Network**: Stable internet connection for API calls

### Recommended Development Setup
- **RAM**: 32GB for full stack development
- **CPU**: 8+ cores for optimal performance
- **Storage**: SSD with 50GB+ free space
- **GPU**: Optional, for local AI model inference

## Core Dependencies Installation

### 1. Python Environment Setup

```bash
# Install Python 3.11+ (Ubuntu/Debian)
sudo apt update
sudo apt install python3.11 python3.11-pip python3.11-venv python3.11-dev

# Install Python 3.11+ (macOS with Homebrew)
brew install python@3.11

# Install Python 3.11+ (Windows - Download from python.org)
# Or use Windows Subsystem for Linux (WSL2)
```

### 2. Virtual Environment Creation

```bash
# Create project directory
mkdir -p ~/development/museQuill.ink
cd ~/development/museQuill.ink

# Clone repository
git clone https://github.com/alexh-scrt/museQuill.ink.git .

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Project Dependencies Installation

```bash
# Install dependencies (choose based on your needs)

# Option 1: Minimal installation (basic functionality)
pip install -r requirements-base.txt

# Option 2: AI capabilities (recommended for development)
pip install -r requirements-ai.txt

# Option 3: Full installation (all features + development tools)
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, langchain, openai; print('Core dependencies installed successfully')"
```

## Database Setup

### PostgreSQL Setup (Primary Database)

#### Ubuntu/Debian Installation
```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib postgresql-client

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql -c "CREATE DATABASE museQuill;"
sudo -u postgres psql -c "CREATE USER museQuill_user WITH PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE museQuill TO museQuill_user;"
sudo -u postgres psql -c "ALTER USER museQuill_user CREATEDB;"
```

#### macOS Installation
```bash
# Install PostgreSQL via Homebrew
brew install postgresql@15
brew services start postgresql@15

# Create database and user
createdb museQuill
psql museQuill -c "CREATE USER museQuill_user WITH PASSWORD 'your_secure_password';"
psql museQuill -c "GRANT ALL PRIVILEGES ON DATABASE museQuill TO museQuill_user;"
```

#### Windows Installation
```bash
# Download and install PostgreSQL from https://www.postgresql.org/download/windows/
# Or use WSL2 with Ubuntu instructions above

# After installation, use pgAdmin or command line:
psql -U postgres
CREATE DATABASE museQuill;
CREATE USER museQuill_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE museQuill TO museQuill_user;
```

### MongoDB Setup (Document Storage)

#### Ubuntu/Debian Installation
```bash
# Import MongoDB public key
curl -fsSL https://pgp.mongodb.com/server-7.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install MongoDB
sudo apt update
sudo apt install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod

# Create database and collections
mongosh --eval "use museQuill"
mongosh museQuill --eval "db.createCollection('books')"
mongosh museQuill --eval "db.createCollection('book_files')"
mongosh museQuill --eval "db.createCollection('book_metadata')"
```

#### macOS Installation
```bash
# Install MongoDB via Homebrew
brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0

# Verify installation
mongosh --eval "db.adminCommand('ismaster')"
```

#### Windows Installation
```bash
# Download MongoDB Community Server from https://www.mongodb.com/try/download/community
# Or use WSL2 with Ubuntu instructions above

# After installation, start MongoDB service and create database:
mongosh
use museQuill
db.createCollection('books')
db.createCollection('book_files')
db.createCollection('book_metadata')
```

## Vector Storage Setup

### Option 1: ChromaDB (Recommended for Development)

```bash
# ChromaDB is installed with requirements and runs embedded
# No additional setup required - it will create local files

# Create data directory
mkdir -p ./chroma_data
chmod 755 ./chroma_data

# Test ChromaDB
python -c "
import chromadb
client = chromadb.PersistentClient('./chroma_data')
print('ChromaDB setup successful')
"
```

### Option 2: Pinecone (Cloud Solution)

```bash
# Sign up at https://www.pinecone.io/
# Get your API key and environment from the dashboard

# Test Pinecone connection
python -c "
import pinecone
pinecone.init(api_key='your-api-key', environment='your-environment')
print('Pinecone connection successful')
"
```

### Option 3: Qdrant (Self-Hosted)

#### Docker Installation (Recommended)
```bash
# Install Docker if not already installed
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Run Qdrant container
docker run -d --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Test Qdrant connection
curl http://localhost:6333/collections
```

#### Manual Installation
```bash
# Download and install Qdrant
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
sudo mv qdrant /usr/local/bin/

# Create configuration
mkdir -p ~/.config/qdrant
cat > ~/.config/qdrant/config.yaml << EOF
service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334

storage:
  storage_path: ./qdrant_data
EOF

# Start Qdrant
qdrant --config ~/.config/qdrant/config.yaml
```

### Option 4: FAISS (Local, CPU-based)

```bash
# FAISS is installed with requirements for CPU usage
# For GPU support, install faiss-gpu instead

# Test FAISS
python -c "
import faiss
import numpy as np
d = 128  # dimension
nb = 1000  # database size
nq = 10   # nb of queries
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
index = faiss.IndexFlatL2(d)
index.add(xb)
print(f'FAISS index built with {index.ntotal} vectors')
"
```

## Knowledge Graph Setup (Neo4j)

### Docker Installation (Recommended)

```bash
# Create Neo4j data directory
mkdir -p ./neo4j_data

# Run Neo4j container
docker run -d \
  --name neo4j-museQuill \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_secure_password \
  -e NEO4J_PLUGINS='["apoc"]' \
  -v ./neo4j_data:/data \
  neo4j:5.15-community

# Wait for Neo4j to start (check logs)
docker logs -f neo4j-museQuill

# Access Neo4j browser at http://localhost:7474
# Login with username: neo4j, password: your_secure_password
```

### Manual Installation

#### Ubuntu/Debian Installation
```bash
# Import Neo4j signing key
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -

# Add Neo4j repository
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install neo4j

# Configure Neo4j
sudo vim /etc/neo4j/neo4j.conf
# Uncomment and modify:
# dbms.default_listen_address=0.0.0.0
# dbms.connector.bolt.listen_address=:7687
# dbms.connector.http.listen_address=:7474

# Set initial password
sudo neo4j-admin dbms set-initial-password your_secure_password

# Start Neo4j service
sudo systemctl start neo4j
sudo systemctl enable neo4j
```

#### macOS Installation
```bash
# Install Neo4j via Homebrew
brew install neo4j

# Start Neo4j
brew services start neo4j

# Set initial password
neo4j-admin dbms set-initial-password your_secure_password
```

### Neo4j Configuration for MuseQuill

```cypher
# Connect to Neo4j browser (http://localhost:7474) and run:

# Create constraints for unique identifiers
CREATE CONSTRAINT book_id_unique FOR (b:Book) REQUIRE b.book_id IS UNIQUE;
CREATE CONSTRAINT character_id_unique FOR (c:Character) REQUIRE c.character_id IS UNIQUE;
CREATE CONSTRAINT chapter_id_unique FOR (ch:Chapter) REQUIRE ch.chapter_id IS UNIQUE;

# Create indexes for performance
CREATE INDEX book_title_index FOR (b:Book) ON (b.title);
CREATE INDEX character_name_index FOR (c:Character) ON (c.name);
CREATE INDEX chapter_number_index FOR (ch:Chapter) ON (ch.chapter_number);

# Create sample nodes for testing
CREATE (b:Book {book_id: 'test-book-1', title: 'Test Book', genre: 'Science Fiction'});
CREATE (c:Character {character_id: 'test-char-1', name: 'Test Character', role: 'protagonist'});
CREATE (ch:Chapter {chapter_id: 'test-chapter-1', chapter_number: 1, title: 'The Beginning'});

# Create relationships
MATCH (b:Book {book_id: 'test-book-1'}), (c:Character {character_id: 'test-char-1'})
CREATE (b)-[:HAS_CHARACTER]->(c);

MATCH (b:Book {book_id: 'test-book-1'}), (ch:Chapter {chapter_id: 'test-chapter-1'})
CREATE (b)-[:HAS_CHAPTER]->(ch);
```

## Caching Setup (Redis)

### Docker Installation (Recommended)

```bash
# Run Redis container
docker run -d \
  --name redis-museQuill \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:7-alpine redis-server --appendonly yes

# Test Redis connection
docker exec -it redis-museQuill redis-cli ping
# Should return: PONG
```

### Manual Installation

#### Ubuntu/Debian Installation
```bash
# Install Redis
sudo apt update
sudo apt install redis-server

# Configure Redis
sudo vim /etc/redis/redis.conf
# Modify:
# supervised systemd
# maxmemory 256mb
# maxmemory-policy allkeys-lru

# Start Redis service
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis
redis-cli ping
# Should return: PONG
```

#### Windows Installation
```bash
# Download Redis for Windows from https://github.com/microsoftarchive/redis/releases
# Or use WSL2 with Ubuntu instructions above

# Alternative: Use Redis in Docker (recommended for Windows)
docker run -d --name redis-museQuill -p 6379:6379 redis:7-alpine
```

### Redis Configuration for MuseQuill

```bash
# Connect to Redis CLI
redis-cli

# Test basic operations
SET test_key "MuseQuill Redis Test"
GET test_key
DEL test_key

# Configure Redis for MuseQuill (optional optimizations)
CONFIG SET maxmemory 512mb
CONFIG SET maxmemory-policy allkeys-lru
CONFIG REWRITE

# Exit Redis CLI
exit
```

## Search APIs Configuration

### Tavily Search API

```bash
# 1. Sign up at https://tavily.com/
# 2. Get your API key from the dashboard
# 3. Test the API

curl -X POST "https://api.tavily.com/search" \
  -H "Content-Type: application/json" \
  -d '{
    "api_key": "your-tavily-api-key",
    "query": "test search",
    "max_results": 5
  }'
```

### Brave Search API

```bash
# 1. Sign up at https://brave.com/search/api/
# 2. Get your API key from the dashboard
# 3. Test the API

curl -X GET "https://api.search.brave.com/res/v1/web/search?q=test%20search&count=5" \
  -H "Accept: application/json" \
  -H "Accept-Encoding: gzip" \
  -H "X-Subscription-Token: your-brave-api-key"
```

### DuckDuckGo (No API Key Required)

```bash
# DuckDuckGo search is integrated via the duckduckgo-search package
# Test the integration

python -c "
from duckduckgo_search import DDGS
with DDGS() as ddgs:
    results = list(ddgs.text('test search', max_results=5))
    print(f'Found {len(results)} results')
    print('DuckDuckGo search working')
"
```

## Application Configuration

### Environment Variables Setup

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your specific configuration
vim .env
```

### Complete .env Configuration

```env
# ===================================
# MuseQuill.Ink Configuration
# ===================================

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL_MUSE=gpt-4
OPENAI_MODEL_SCRIBE=gpt-4o
OPENAI_MODEL_SCHOLAR=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Database Configuration
DATABASE_URL=postgresql://museQuill_user:your_secure_password@localhost:5432/museQuill
REDIS_URL=redis://localhost:6379/0

# MongoDB Configuration (Optional - for book storage)
MONGODB_URL=mongodb://localhost:27017/museQuill
MONGODB_DATABASE=museQuill

# Knowledge Graph Configuration (Optional - Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password
NEO4J_DATABASE=museQuill

# Vector Storage Configuration (Choose one)
# Option 1: ChromaDB (Local, recommended for development)
CHROMA_PERSIST_DIRECTORY=./chroma_data

# Option 2: Pinecone (Cloud)
# PINECONE_API_KEY=your-pinecone-api-key
# PINECONE_ENVIRONMENT=your-pinecone-environment

# Option 3: Qdrant (Self-hosted)
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=your-qdrant-api-key  # Optional, if authentication enabled

# Research APIs
TAVILY_API_KEY=your-tavily-api-key
BRAVE_API_KEY=your-brave-search-api-key

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=your-very-secure-secret-key-here-change-in-production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# CORS Configuration (for frontend development)
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:5500"]

# Feature Flags
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_VECTOR_SEARCH=true
ENABLE_RESEARCH_APIS=true
ENABLE_BOOK_STORAGE=true
ENABLE_AGENT_METRICS=true

# Performance Settings
MAX_CONCURRENT_AGENTS=5
AGENT_TIMEOUT_SECONDS=300
VECTOR_SEARCH_LIMIT=20
KNOWLEDGE_GRAPH_QUERY_LIMIT=100

# Storage Settings
MAX_BOOK_SIZE_MB=50
MAX_UPLOAD_SIZE_MB=10
FILE_STORAGE_PATH=./storage
TEMP_STORAGE_PATH=./temp

# Monitoring and Logging
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_FILE_PATH=./logs/musequill.log
LOG_ROTATION_SIZE=100MB
LOG_RETENTION_DAYS=30
```

### Application Bootstrap

```bash
# Create necessary directories
mkdir -p logs storage temp chroma_data

# Set proper permissions
chmod 755 logs storage temp chroma_data

# Initialize the application
python bootstrap.py

# Run database migrations (if using Alembic)
# alembic upgrade head

# Test the configuration
python -c "
from musequill.config.settings import Settings
settings = Settings()
print('Configuration loaded successfully')
print(f'Environment: {settings.ENVIRONMENT}')
print(f'Database URL: {settings.DATABASE_URL[:20]}...')
"
```

## Development Tools Setup

### Code Quality Tools

```bash
# Install development dependencies (if not already installed)
pip install -r requirements.txt

# Setup pre-commit hooks
pre-commit install

# Run code formatting
black musequill/
isort musequill/

# Run linting
ruff musequill/
mypy musequill/

# Run tests
pytest tests/ -v --cov=musequill
```

### IDE Configuration

#### VS Code Setup

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": false,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.provider": "isort",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true,
    "*.egg-info": true
  }
}
```

#### PyCharm Setup

```bash
# Configure PyCharm interpreter
# File -> Settings -> Project -> Python Interpreter
# Add Local Interpreter -> Existing Environment
# Point to: ./venv/bin/python

# Configure code style
# File -> Settings -> Editor -> Code Style -> Python
# Set line length to 88 (Black default)
# Enable "Optimize imports on the fly"
```

## Verification and Testing

### System Health Check

```bash
# Start all services and run health check
python api.py &
API_PID=$!

# Wait for API to start
sleep 5

# Test health endpoint
curl http://localhost:8000/api/health

# Test enum endpoint
curl http://localhost:8000/api/enums

# Stop API
kill $API_PID
```

### Database Connectivity Tests

```bash
# Test PostgreSQL connection
python -c "
import psycopg2
from musequill.config.settings import Settings
settings = Settings()
try:
    conn = psycopg2.connect(settings.DATABASE_URL)
    print('✅ PostgreSQL connection successful')
    conn.close()
except Exception as e:
    print(f'❌ PostgreSQL connection failed: {e}')
"

# Test MongoDB connection (if enabled)
python -c "
from pymongo import MongoClient
from musequill.config.settings import Settings
settings = Settings()
try:
    client = MongoClient(settings.MONGODB_URL)
    client.admin.command('ismaster')
    print('✅ MongoDB connection successful')
    client.close()
except Exception as e:
    print(f'❌ MongoDB connection failed: {e}')
"

# Test Redis connection
python -c "
import redis
from musequill.config.settings import Settings
settings = Settings()
try:
    r = redis.from_url(settings.REDIS_URL)
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
"

# Test Neo4j connection (if enabled)
python -c "
from neo4j import GraphDatabase
from musequill.config.settings import Settings
settings = Settings()
try:
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run('RETURN 1 as test')
        record = result.single()
        if record['test'] == 1:
            print('✅ Neo4j connection successful')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
"
```

### AI Integration Tests

```bash
# Test OpenAI API connection
python -c "
from openai import OpenAI
from musequill.config.settings import Settings
settings = Settings()
try:
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'Test'}],
        max_tokens=10
    )
    print('✅ OpenAI API connection successful')
    print(f'Response: {response.choices[0].message.content}')
except Exception as e:
    print(f'❌ OpenAI API connection failed: {e}')
"

# Test agent factory
python -c "
from musequill.agents.factory import get_agent_factory
from musequill.core.openai_client import OpenAIClient
from musequill.config.settings import Settings
try:
    settings = Settings()
    openai_client = OpenAIClient(api_key=settings.OPENAI_API_KEY)
    factory = get_agent_factory(openai_client)
    print('✅ Agent factory initialization successful')
    print(f'Factory ready: {factory is not None}')
except Exception as e:
    print(f'❌ Agent factory initialization failed: {e}')
"
```

### Research API Tests

```bash
# Test Tavily API (if configured)
python -c "
from musequill.config.settings import Settings
import requests
settings = Settings()
if hasattr(settings, 'TAVILY_API_KEY') and settings.TAVILY_API_KEY:
    try:
        response = requests.post(
            'https://api.tavily.com/search',
            json={
                'api_key': settings.TAVILY_API_KEY,
                'query': 'test',
                'max_results': 1
            }
        )
        if response.status_code == 200:
            print('✅ Tavily API connection successful')
        else:
            print(f'❌ Tavily API error: {response.status_code}')
    except Exception as e:
        print(f'❌ Tavily API connection failed: {e}')
else:
    print('⚠️  Tavily API key not configured')
"

# Test DuckDuckGo search
python -c "
try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text('test', max_results=1))
        print('✅ DuckDuckGo search successful')
except Exception as e:
    print(f'❌ DuckDuckGo search failed: {e}')
"
```

## Docker Setup (Alternative)

For a simplified setup, you can use Docker Compose to run all services:

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: museQuill
      POSTGRES_USER: museQuill_user
      POSTGRES_PASSWORD: your_secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U museQuill_user -d museQuill"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # MongoDB (Optional)
  mongodb:
    image: mongo:7
    environment:
      MONGO_INITDB_DATABASE: museQuill
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Neo4j Knowledge Graph (Optional)
  neo4j:
    image: neo4j:5.15-community
    environment:
      NEO4J_AUTH: neo4j/your_secure_password
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "your_secure_password", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Qdrant Vector Database (Optional)
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  postgres_data:
  redis_data:
  mongodb_data:
  neo4j_data:
  qdrant_data:
```

### Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Stop and remove volumes (data will be lost)
docker-compose down -v

# Update services
docker-compose pull
docker-compose up -d
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Python Version Issues

```bash
# Error: Python 3.11+ required
# Solution: Install correct Python version
pyenv install 3.11.7
pyenv local 3.11.7

# Or use conda
conda create -n musequill python=3.11
conda activate musequill
```

#### 2. PostgreSQL Connection Issues

```bash
# Error: Connection refused
# Solution: Check PostgreSQL status
sudo systemctl status postgresql

# Fix: Start PostgreSQL
sudo systemctl start postgresql

# Error: Authentication failed
# Solution: Reset password
sudo -u postgres psql
ALTER USER museQuill_user PASSWORD 'new_password';
```

#### 3. Redis Memory Issues

```bash
# Error: Out of memory
# Solution: Increase Redis memory limit
redis-cli CONFIG SET maxmemory 1gb

# Or edit Redis config
sudo vim /etc/redis/redis.conf
# Add: maxmemory 1gb
sudo systemctl restart redis
```

#### 4. Neo4j Connection Issues

```bash
# Error: Authentication failed
# Solution: Reset Neo4j password
docker exec -it neo4j-museQuill neo4j-admin dbms set-initial-password new_password

# Error: Service unavailable
# Solution: Check Neo4j logs
docker logs neo4j-museQuill
```

#### 5. OpenAI API Issues

```bash
# Error: Invalid API key
# Solution: Check API key format and billing
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer your-api-key"

# Error: Rate limit exceeded
# Solution: Implement backoff strategy or upgrade plan
```

#### 6. Vector Storage Issues

```bash
# ChromaDB: Permission denied
sudo chown -R $USER:$USER ./chroma_data
chmod -R 755 ./chroma_data

# Pinecone: Environment issues
# Check environment name in Pinecone dashboard

# Qdrant: Service not responding
docker restart qdrant
curl http://localhost:6333/health
```

### Performance Optimization

```bash
# 1. Database optimization
# PostgreSQL: Tune configuration
sudo vim /etc/postgresql/15/main/postgresql.conf
# shared_buffers = 256MB
# effective_cache_size = 1GB

# 2. Redis optimization
# Enable persistence and compression
redis-cli CONFIG SET save "900 1 300 10 60 10000"

# 3. Application optimization
# Use environment variables for production
export UVICORN_WORKERS=4
export UVICORN_HOST=0.0.0.0
export UVICORN_PORT=8000

# 4. Monitor resource usage
htop
docker stats
```

### Getting Help

#### Log Analysis

```bash
# Application logs
tail -f logs/musequill.log

# Database logs
sudo tail -f /var/log/postgresql/postgresql-15-main.log

# Docker logs
docker-compose logs -f service_name
```

#### Health Monitoring

```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== MuseQuill.Ink Health Check ==="
echo "Timestamp: $(date)"
echo ""

# Check API
echo "API Health:"
curl -s http://localhost:8000/api/health | jq '.status' || echo "❌ API Down"
echo ""

# Check databases
echo "Database Status:"
pg_isready -h localhost -p 5432 -U museQuill_user && echo "✅ PostgreSQL" || echo "❌ PostgreSQL"
redis-cli ping > /dev/null && echo "✅ Redis" || echo "❌ Redis"
docker exec mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1 && echo "✅ MongoDB" || echo "❌ MongoDB"
docker exec neo4j-museQuill cypher-shell -u neo4j -p your_secure_password "RETURN 1" > /dev/null 2>&1 && echo "✅ Neo4j" || echo "❌ Neo4j"
echo ""

# Check disk space
echo "Disk Usage:"
df -h | grep -E "(/$|/data)"
echo ""
EOF

chmod +x monitor.sh
./monitor.sh
```

This comprehensive setup guide should help you configure a complete development environment for MuseQuill.Ink. Remember to:

1. **Secure your environment**: Use strong passwords and API keys
2. **Regular backups**: Backup your databases and configuration
3. **Monitor resources**: Keep an eye on memory, disk, and API usage
4. **Stay updated**: Regularly update dependencies and services
5. **Test thoroughly**: Verify each component before proceeding

For additional help, refer to the project documentation in the `docs/` directory or create an issue in the GitHub repository. ping
# Should return: PONG
```

#### macOS Installation
```bash
# Install Redis via Homebrew
brew install redis

# Start Redis service
brew services start redis

# Test Redis
redis-cli