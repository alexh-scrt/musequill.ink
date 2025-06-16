# MuseQuill.Ink

**AI-Powered Creative Writing Platform**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-purple.svg)](https://python.langchain.com/)

> **"AI-Assisted Creative Partnership, Not Replacement"**

MuseQuill.Ink is an intelligent creative writing platform that enhances human creativity through specialized AI assistants. Unlike generic AI writing tools, MuseQuill provides a comprehensive workflow designed specifically for fiction authors, from initial inspiration to published manuscript.

## ✨ Key Features

### 🎭 Three Specialized AI Assistants

- **🎨 The Muse** (GPT-4) - Story architect and creative visionary
- **✍️ The Scribe** (GPT-4o) - Writing partner and prose specialist  
- **🔍 The Scholar** (GPT-3.5-turbo) - Research assistant and fact-checker

### 🧠 Intelligent Memory System

- **Persistent Story Memory** - Maintains context across entire manuscripts
- **Character Consistency** - Tracks character development and prevents contradictions
- **World-Building Coherence** - Ensures consistent fictional world rules
- **Plot Tracking** - Monitors story arcs and prevents plot holes

### 🔬 Integrated Research Capabilities

- **Multi-Source Research** - Tavily, Brave Search, DuckDuckGo integration
- **Real-Time Fact Checking** - Automated verification during writing
- **Citation Tracking** - Maintains research trails for authenticity
- **Cultural Sensitivity** - Ensures respectful representation

### 📚 Comprehensive Writing Workflow

1. **Inspiration & Ideation** - AI-assisted concept development
2. **Planning & Structure** - Intelligent outline and character creation
3. **Research & Context** - Automated fact-gathering and verification
4. **Drafting** - Collaborative writing assistance
5. **Revision & Refinement** - Style and consistency optimization
6. **Editing & Polish** - Final quality assurance

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Redis (for caching and sessions)
- PostgreSQL or MongoDB (for data persistence)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alexh-scrt/museQuill.ink.git
   cd museQuill.ink
   ```

2. **Install dependencies**
   ```bash
   # Basic installation
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
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
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

# Research APIs
TAVILY_API_KEY=your-tavily-key
BRAVE_API_KEY=your-brave-search-key

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
SECRET_KEY=your-secret-key-here
```

## 🏗️ Architecture

MuseQuill.Ink follows a modular architecture designed for scalability and maintainability:

```mermaid
graph TB
    subgraph "User Interface Layer"
        WEB[Web Application]
        API[REST API]
        WS[WebSocket Streams]
    end
    
    subgraph "AI Orchestration Layer"
        MUSE[Muse Agent]
        SCRIBE[Scribe Agent]
        SCHOLAR[Scholar Agent]
        COORD[Agent Coordinator]
    end
    
    subgraph "Intelligence Layer"
        GPT4[OpenAI GPT-4]
        GPT4O[OpenAI GPT-4o]
        GPT35[OpenAI GPT-3.5-turbo]
        EMB[OpenAI Embeddings]
    end
    
    subgraph "Memory Layer"
        KG[Knowledge Graph]
        VS[Vector Store]
        CACHE[Redis Cache]
        FILES[File Storage]
    end
    
    subgraph "Research Layer"
        TAVILY[Tavily Search]
        BRAVE[Brave Search]
        DDG[DuckDuckGo]
        SYNTH[Research Synthesizer]
    end
```

### Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **AI Framework**: LangChain, LangGraph
- **AI Models**: OpenAI GPT-4, GPT-4o, GPT-3.5-turbo
- **Vector Storage**: Pinecone, ChromaDB, Qdrant, FAISS
- **Knowledge Graph**: Neo4j
- **Caching**: Redis
- **Database**: PostgreSQL, MongoDB support
- **Research APIs**: Tavily, Brave Search, DuckDuckGo

## 📁 Project Structure

```
museQuill.ink/
├── musequill/                 # Core application
│   ├── agents/               # AI agent implementations
│   │   ├── muse.py          # Creative architect agent
│   │   ├── scribe.py        # Writing assistant agent
│   │   └── scholar.py       # Research agent
│   ├── api/                 # API layer
│   ├── core/                # Core utilities
│   ├── models/              # Data models and presets
│   ├── config/              # Configuration management
│   └── workers/             # Background processing
├── docs/                    # Documentation
│   ├── architecture.md     # System architecture
│   ├── AI Agent with LangGraph/  # LangGraph guides
│   ├── Knowledge Graphs/    # Knowledge graph tutorials
│   └── LLM Semantic Search/ # Search implementation
├── requirements*.txt        # Python dependencies
├── api.py                  # FastAPI application entry
└── README.md               # This file
```

## 🔧 Development

### Setting up Development Environment

1. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

4. **Code formatting**
   ```bash
   black musequill/
   isort musequill/
   ruff musequill/
   ```

### Development Tools

- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Code Quality**: black, isort, mypy, ruff
- **Pre-commit**: Automated code quality checks
- **Documentation**: mkdocs with material theme

## 🎯 Roadmap

### Phase 1: Foundation (Months 1-3) ✅
- [x] Core AI agent framework
- [ ] Basic memory system (knowledge graph + vector store)
- [x] Simple web interface for story creation [work in progress]
- [ ] OpenAI integration with cost management

### Phase 2: Core Features (Months 4-6) 🚧
- [ ] Full Muse, Scribe, Scholar implementation
- [ ] Research integration (Tavily, Brave, DuckDuckGo)
- [ ] Advanced memory management and context awareness
- [ ] Real-time collaborative writing interface

### Phase 3: Enhancement (Months 7-9) 📋
- [ ] Advanced style and voice learning
- [ ] Multi-project management
- [ ] Collaboration features for co-authors
- [ ] Mobile applications

### Phase 4: Scale (Months 10-12) 🎯
- [ ] Performance optimization
- [ ] Advanced analytics and insights
- [ ] Publishing integration
- [ ] Enterprise features for teams

## 📊 Success Metrics

### Author Success Metrics
- **Writing Velocity**: 50% increase in words per session
- **Quality Consistency**: Reduced plot holes and character inconsistencies
- **Research Integration**: 80% reduction in research time
- **Creative Satisfaction**: High author satisfaction with AI partnership

### Technical Success Metrics
- **Response Time**: <2s for most AI suggestions
- **Accuracy Rate**: >95% factual accuracy in research
- **System Reliability**: 99.9% uptime
- **Cost Efficiency**: Optimal model selection for each task

## 🎮 Usage Examples

### Starting a New Story

```python
from musequill.agents import get_agent_factory

# Initialize the AI assistant team
factory = get_agent_factory()
muse = factory.create_muse()
scribe = factory.create_scribe()
scholar = factory.create_scholar()

# Begin creative collaboration
concept = await muse.develop_concept("time travel romance")
research = await scholar.research_topic("time travel theories")
opening = await scribe.write_opening_scene(concept, research)
```

### API Usage

```bash
# Create a new book project
curl -X POST "http://localhost:8000/api/v1/books" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Novel",
    "genre": "SCIENCE_FICTION",
    "target_length": "STANDARD_NOVEL",
    "writing_style": "LITERARY"
  }'

# Get writing assistance
curl -X POST "http://localhost:8000/api/v1/writing/assist" \
  -H "Content-Type: application/json" \
  -d '{
    "book_id": "uuid-here",
    "context": "opening scene",
    "request": "help me write a compelling first paragraph"
  }'
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Run code quality checks: `pre-commit run --all-files`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing the foundational language models
- **LangChain** team for the excellent AI orchestration framework
- **FastAPI** community for the robust web framework
- **Neo4j** for graph database capabilities
- All the amazing **open-source contributors** who make this possible

## 📞 Support

- **Documentation**: [museQuill.ink](https://musequill.ink)

---

**Built with ❤️ for writers who dream of better tools**

*MuseQuill.Ink - Where Human Creativity Meets AI Intelligence*