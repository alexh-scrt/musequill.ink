# MuseQuill.Ink

**AI-Powered Creative Writing Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.20+-orange.svg)](https://langchain-ai.github.io/langgraph/)

> **"AI-Assisted Creative Partnership, Not Replacement"**

MuseQuill.Ink is an intelligent creative writing platform that enhances human creativity through specialized AI assistants. Unlike generic AI writing tools, MuseQuill provides a comprehensive workflow designed specifically for fiction authors, from initial inspiration to published manuscript.

## ✨ Key Features

### 🎭 Three Specialized AI Assistants

- **🎨 The Muse** (GPT-4) - Story architect and creative visionary
  - Develops story concepts and themes
  - Creates comprehensive outlines and story structures
  - Provides creative direction and plot development
  
- **✍️ The Scribe** (GPT-4o) - Writing partner and prose specialist
  - Assists with chapter writing and scene development
  - Maintains consistency in voice and style
  - Provides real-time writing suggestions and improvements
  
- **🔍 The Scholar** (GPT-3.5-turbo) - Research assistant and fact-checker
  - Conducts multi-source research via Tavily, Brave Search, DuckDuckGo
  - Validates facts and ensures authenticity
  - Provides cultural context and sensitivity checks

### 🧠 Intelligent Memory System

- **Knowledge Graph Integration** (Neo4j) - Maintains complex relationships between characters, plot elements, and world-building
- **Vector Storage** (Pinecone/ChromaDB/Qdrant/FAISS) - Semantic search and context retrieval
- **Persistent Story Memory** - Maintains context across entire manuscripts
- **Character Consistency** - Tracks character development and prevents contradictions
- **World-Building Coherence** - Ensures consistent fictional world rules
- **Plot Tracking** - Monitors story arcs and prevents plot holes

### 🔄 LangGraph Orchestration

- **Multi-Agent Workflows** - Coordinated collaboration between specialized agents
- **State Management** - Persistent state across writing sessions
- **Conditional Routing** - Intelligent decision-making for agent workflows
- **Human-in-the-Loop** - Author maintains creative control at every step

### 🔬 Integrated Research Capabilities

- **Multi-Source Research** - Tavily, Brave Search, DuckDuckGo integration
- **Real-Time Fact Checking** - Automated verification during writing
- **Citation Tracking** - Maintains research trails for authenticity
- **Cultural Sensitivity** - Ensures respectful representation

### 📚 Comprehensive Writing Workflow

1. **Inspiration & Ideation** - AI-assisted concept development
2. **Planning & Structure** - Intelligent outline and character creation
3. **Research & Context** - Automated fact-gathering and verification
4. **Drafting** - Collaborative writing assistance with LangGraph orchestration
5. **Revision & Refinement** - Style and consistency optimization
6. **Editing & Polish** - Final quality assurance
7. **Storage & Management** - MongoDB integration for manuscript persistence

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Redis (for caching and sessions)
- PostgreSQL (for application data)
- MongoDB (for book storage, optional)
- Neo4j (for knowledge graphs, optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alexh-scrt/museQuill.ink.git
   cd museQuill.ink
   ```

2. **Install dependencies**
   ```bash
   # Basic installation (minimal dependencies)
   pip install -r requirements-base.txt
   
   # Full AI capabilities
   pip install -r requirements-ai.txt
   
   # Complete installation with development tools
   pip install -r requirements.txt
   ```

3. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Start the application**
   ```bash
   python api.py --host 0.0.0.0 --port 8000
   ```

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL_MUSE=gpt-4
OPENAI_MODEL_SCRIBE=gpt-4o
OPENAI_MODEL_SCHOLAR=gpt-3.5-turbo

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/museQuill
REDIS_URL=redis://localhost:6379/0
MONGODB_URL=mongodb://localhost:27017/museQuill

# Knowledge Graph (Optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=museQuill

# Vector Storage (Choose one)
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-environment
# OR
CHROMA_PERSIST_DIRECTORY=./chroma_data
# OR
QDRANT_URL=http://localhost:6333

# Research APIs
TAVILY_API_KEY=your-tavily-key
BRAVE_API_KEY=your-brave-search-key

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here
```

## 🏗️ Architecture

MuseQuill.Ink follows a modular, agent-based architecture designed for scalability and maintainability:

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[Web Application]
        API[REST API]
        WS[WebSocket Streams]
    end
    
    subgraph "AI Orchestration Layer (LangGraph)"
        COORD[Agent Coordinator]
        MUSE[Muse Agent]
        SCRIBE[Scribe Agent]
        SCHOLAR[Scholar Agent]
        PLANNER[Planning Agent]
        STORER[Book Storage Agent]
    end
    
    subgraph "Intelligence Layer"
        GPT4[OpenAI GPT-4]
        GPT4O[OpenAI GPT-4o]
        GPT35[OpenAI GPT-3.5-turbo]
        EMB[OpenAI Embeddings]
    end
    
    subgraph "Memory & Storage Layer"
        KG[Neo4j Knowledge Graph]
        VS[Vector Store<br/>Pinecone/ChromaDB/Qdrant]
        CACHE[Redis Cache]
        MONGO[MongoDB<br/>Book Storage]
        POSTGRES[PostgreSQL<br/>App Data]
    end
    
    subgraph "Research Layer"
        TAVILY[Tavily Search]
        BRAVE[Brave Search]
        DDG[DuckDuckGo]
    end
    
    WEB --> API
    API --> COORD
    WS --> COORD
    
    COORD --> MUSE
    COORD --> SCRIBE
    COORD --> SCHOLAR
    COORD --> PLANNER
    COORD --> STORER
    
    MUSE --> GPT4
    SCRIBE --> GPT4O
    SCHOLAR --> GPT35
    PLANNER --> GPT4
    
    MUSE --> KG
    SCRIBE --> VS
    SCHOLAR --> CACHE
    STORER --> MONGO
    
    SCHOLAR --> TAVILY
    SCHOLAR --> BRAVE
    SCHOLAR --> DDG
    
    API --> POSTGRES
    COORD --> REDIS
```

### Core Technologies

- **Backend Framework**: FastAPI with Uvicorn
- **AI Orchestration**: LangGraph for multi-agent workflows
- **AI Framework**: LangChain ecosystem
- **AI Models**: OpenAI GPT-4, GPT-4o, GPT-3.5-turbo
- **Vector Storage**: Multiple options (Pinecone, ChromaDB, Qdrant, FAISS)
- **Knowledge Graph**: Neo4j for complex relationships
- **Caching**: Redis for session management and performance
- **Primary Database**: PostgreSQL for application data
- **Document Storage**: MongoDB for manuscript persistence
- **Research APIs**: Tavily, Brave Search, DuckDuckGo

## 📁 Project Structure

```
museQuill.ink/
├── musequill/                 # Core application package
│   ├── agents/               # AI agent implementations
│   │   ├── muse.py          # Creative architect agent
│   │   ├── scribe.py        # Writing assistant agent
│   │   ├── scholar.py       # Research agent
│   │   ├── planning.py      # Story planning agent
│   │   ├── orchestrator.py  # LangGraph workflow orchestration
│   │   ├── factory.py       # Agent factory and management
│   │   └── book_storer/     # Book storage agent
│   ├── api/                 # API layer
│   │   └── model.py         # API data models
│   ├── core/                # Core utilities
│   │   ├── base/           # Base agent classes
│   │   └── openai_client/  # OpenAI client wrapper
│   ├── models/              # Data models and presets
│   │   ├── presets.py      # Genre and writing presets
│   │   ├── planning.py     # Planning data models
│   │   └── subgenre.py     # Subgenre definitions
│   ├── config/              # Configuration management
│   │   ├── settings.py     # Application settings
│   │   └── logging.py      # Logging configuration
│   ├── prompts/             # AI prompts and templates
│   ├── routers/             # FastAPI route handlers
│   ├── workers/             # Background processing
│   └── utils/               # Utility functions
├── docs/                    # Comprehensive documentation
│   ├── architecture.md     # System architecture blueprint
│   ├── AI Agent with LangGraph/  # LangGraph guides and tutorials
│   ├── Knowledge Graphs/    # Neo4j and knowledge graph tutorials
│   └── LLM Semantic Search/ # Vector search implementation guides
├── static/                  # Frontend assets
├── requirements*.txt        # Python dependencies (tiered)
├── api.py                  # FastAPI application entry point
├── bootstrap.py            # Application bootstrap
├── .env.example            # Environment configuration template
└── README.md               # This file
```

## 🔧 Development

### Setting up Development Environment

1. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

2. **Run tests**
   ```bash
   pytest tests/ -v
   ```

3. **Code formatting**
   ```bash
   black musequill/
   isort musequill/
   ruff musequill/
   ```

### Development Tools

- **Testing**: pytest, pytest-asyncio, pytest-cov, pytest-mock
- **Code Quality**: black, isort, mypy, ruff
- **Pre-commit**: Automated code quality checks
- **Documentation**: mkdocs with material theme and mermaid diagrams
- **Debugging**: Rich console output and structlog

## 🎯 Development Roadmap

### Phase 1: Foundation ✅
- [x] Core AI agent framework with base classes
- [x] LangGraph orchestration system
- [x] FastAPI backend with comprehensive API endpoints
- [x] OpenAI integration with cost tracking
- [x] Multi-tier dependency system (requirements-*.txt)
- [x] Configuration management and environment setup

### Phase 2: Core Features 🚧
- [ ] Complete Muse, Scribe, Scholar agent implementations
- [ ] Knowledge graph integration (Neo4j)
- [ ] Vector storage implementation (multiple backends)
- [ ] Research integration (Tavily, Brave, DuckDuckGo)
- [ ] Real-time collaborative writing interface
- [ ] Advanced memory management and context awareness

### Phase 3: Enhancement 📋
- [ ] Advanced style and voice learning
- [ ] Multi-project management
- [ ] Collaboration features for co-authors
- [ ] Mobile applications
- [ ] Publishing integration workflows
- [ ] Advanced analytics and writing insights

### Phase 4: Scale 🎯
- [ ] Performance optimization and caching strategies
- [ ] Enterprise features for teams
- [ ] Custom model fine-tuning
- [ ] Advanced workflow customization
- [ ] Marketplace for writing templates and agents

## 📊 Success Metrics

### Author Success Metrics
- **Writing Velocity**: 50% increase in words per session
- **Quality Consistency**: Reduced plot holes and character inconsistencies
- **Research Integration**: 80% reduction in research time
- **Creative Satisfaction**: High author satisfaction with AI partnership

### Technical Success Metrics
- **Response Time**: <2s for most AI suggestions
- **Accuracy Rate**: >95% factual accuracy in research
- **System Reliability**: 99.9% uptime target
- **Cost Efficiency**: Optimal model selection for each task

## 🎮 Usage Examples

### API Usage

```bash
# Health check
curl http://localhost:8000/api/health

# Get available enums for form population
curl http://localhost:8000/api/enums

# Create a new book project
curl -X POST "http://localhost:8000/api/v1/books" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "The Time Travelers Dilemma",
    "genre": "SCIENCE_FICTION",
    "subgenre": "time_travel",
    "target_length": "STANDARD_NOVEL",
    "structure": "HEROS_JOURNEY",
    "writing_style": "LITERARY",
    "description": "A physicist discovers time travel but faces moral dilemmas"
  }'

# Start AI planning for a book
curl -X POST "http://localhost:8000/api/v1/planning/create-plan" \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "your-book-id",
    "description": "Help me plan a time travel story",
    "genre": "SCIENCE_FICTION"
  }'
```

### Python Usage

```python
from musequill.agents import get_agent_factory
from musequill.models.planning import PlanningRequest

# Initialize the AI assistant team
factory = get_agent_factory()
planning_agent = factory.create_planning_agent()

# Create a comprehensive book plan
request = PlanningRequest(
    description="A story about time travel and its consequences",
    genre=GenreType.SCIENCE_FICTION,
    target_length=BookLength.STANDARD_NOVEL
)

plan = await planning_agent.create_complete_plan(request)
print(f"Story Outline: {plan.outline.summary}")
print(f"Chapters: {len(plan.outline.chapters)}")
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`pytest && black . && ruff .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** for the excellent AI framework and LangGraph orchestration
- **OpenAI** for providing the foundational language models
- **FastAPI** for the high-performance web framework
- **Neo4j** for the powerful graph database capabilities
- **The Open Source Community** for the amazing tools and libraries

## 📞 Support

- **Documentation**: Comprehensive guides in the `docs/` directory
- **Issues**: [GitHub Issues](https://github.com/alexh-scrt/museQuill.ink/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alexh-scrt/museQuill.ink/discussions)

---

**MuseQuill.Ink** - Where AI meets creativity to help authors write better books. 🖋️✨