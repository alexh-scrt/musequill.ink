# Knowledge Graph Construction from Documents using Neo4j
## A Complete Developer Manual

### Table of Contents
1. [Overview and Architecture](#overview-and-architecture)
2. [Environment Setup and Dependencies](#environment-setup-and-dependencies)
3. [Document Processing Pipeline](#document-processing-pipeline)
4. [Text Chunking Strategy](#text-chunking-strategy)
5. [Neo4j Graph Database Integration](#neo4j-graph-database-integration)
6. [Vector Embeddings and Similarity Search](#vector-embeddings-and-similarity-search)
7. [RAG Implementation with LangChain](#rag-implementation-with-langchain)
8. [Best Practices and Optimization](#best-practices-and-optimization)

---

## Overview and Architecture

This manual explains how to construct Knowledge Graphs from text documents using Neo4j as the graph database backend. The approach demonstrated processes structured documents (specifically SEC Form 10-K filings) and creates a searchable knowledge graph with vector-based similarity search capabilities.

### Core Architecture Components

The system consists of several key components working together:

1. **Document Processing Layer**: Handles document ingestion and parsing
2. **Text Chunking Engine**: Splits large documents into manageable segments
3. **Graph Database Layer**: Neo4j for storing nodes and relationships
4. **Vector Embedding System**: OpenAI embeddings for semantic search
5. **RAG (Retrieval-Augmented Generation)**: LangChain integration for question-answering

### Data Flow Architecture

```
Raw Documents → Text Extraction → Chunking → Graph Nodes → Vector Embeddings → Search Index → RAG System
```

---

## Environment Setup and Dependencies

### Required Libraries and Imports

The implementation relies on several key Python libraries:

```python
# Environment and configuration
from dotenv import load_dotenv
import os
import json
import textwrap
import warnings

# LangChain ecosystem
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
```

### Environment Configuration

The system requires several environment variables for proper operation:

- **Neo4j Connection**: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- **OpenAI Integration**: `OPENAI_API_KEY`, `OPENAI_BASE_URL`

### Key Constants Definition

```python
VECTOR_INDEX_NAME = 'form_10k_chunks'      # Name of the vector index
VECTOR_NODE_LABEL = 'Chunk'                # Label for chunk nodes
VECTOR_SOURCE_PROPERTY = 'text'            # Property containing text content
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding' # Property storing embeddings
```

These constants define the structure of how data is organized in the Neo4j database and ensure consistency across the application.

---

## Document Processing Pipeline

### Document Structure Analysis

The implementation works with JSON-formatted SEC Form 10-K documents. Each document contains multiple sections:

- `item1`: Business overview
- `item1a`: Risk factors  
- `item7`: Management discussion and analysis
- `item7a`: Market risk disclosures

### Document Loading and Parsing

```python
first_file_as_object = json.load(open(first_file_name))
```

The system loads JSON files and extracts specific sections. Each document object contains:
- **Content sections**: The actual text content for each item
- **Metadata**: Company identifiers (CIK, CUSIP), names, and source information

### Metadata Extraction

Key metadata fields preserved during processing:
- `names`: Company names and aliases
- `cik`: Central Index Key (SEC identifier)
- `cusip6`: Committee on Uniform Securities Identification Procedures code
- `source`: Original document source URL

---

## Text Chunking Strategy

### Chunking Algorithm Configuration

The system uses LangChain's `RecursiveCharacterTextSplitter` with specific parameters:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000,        # Maximum characters per chunk
    chunk_overlap = 200,      # Overlap between adjacent chunks
    length_function = len,    # Function to measure chunk length
    is_separator_regex = False # Use simple string separators
)
```

### Chunking Parameters Explained

- **chunk_size (2000)**: Optimal balance between context preservation and processing efficiency
- **chunk_overlap (200)**: Ensures continuity across chunk boundaries, preventing loss of context at splits
- **length_function**: Uses character count rather than token count for consistency

### Chunk Processing Function

The `split_form10k_data_from_file()` function demonstrates sophisticated chunk processing:

1. **Iterates through document sections**: Processes specific items (item1, item1a, item7, item7a)
2. **Applies chunking algorithm**: Splits each section using the configured text splitter
3. **Limits chunks per section**: Takes only first 20 chunks to manage processing time
4. **Constructs metadata-rich records**: Each chunk includes:
   - Original text content
   - Section identifier (f10kItem)
   - Sequence information (chunkSeqId)
   - Unique identifier (chunkId)
   - Company metadata (names, cik, cusip6, source)

### Chunk Identification Strategy

Each chunk receives a unique identifier constructed as:
```
{formId}-{item}-chunk{sequenceId:04d}
```

This pattern ensures:
- **Global uniqueness**: No duplicate chunks across documents
- **Hierarchical organization**: Easy identification of source document and section
- **Sequential ordering**: Maintains document structure relationships

---

## Neo4j Graph Database Integration

### Database Connection Setup

The system establishes connection to Neo4j using LangChain's integration:

```python
kg = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD, 
    database=NEO4J_DATABASE
)
```

### Node Creation Strategy

#### Chunk Node Schema

Each document chunk becomes a node with the label `:Chunk` and properties:
- `chunkId`: Unique identifier
- `text`: The actual text content
- `formId`: Source document identifier
- `f10kItem`: Document section
- `chunkSeqId`: Sequence within section
- `names`, `cik`, `cusip6`, `source`: Company metadata

#### MERGE Query Pattern

The system uses MERGE operations to ensure idempotent node creation:

```cypher
MERGE(mergedChunk:Chunk {chunkId: $chunkParam.chunkId})
    ON CREATE SET 
        mergedChunk.names = $chunkParam.names,
        mergedChunk.formId = $chunkParam.formId,
        -- additional properties...
RETURN mergedChunk
```

### Constraint Implementation

To prevent duplicate nodes, the system creates a uniqueness constraint:

```cypher
CREATE CONSTRAINT unique_chunk IF NOT EXISTS 
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
```

This constraint:
- **Ensures data integrity**: Prevents accidental duplicates
- **Improves performance**: Creates an index on chunkId
- **Enables safe re-runs**: MERGE operations won't create duplicates

---

## Vector Embeddings and Similarity Search

### Vector Index Creation

The system creates a specialized vector index for similarity search:

```cypher
CREATE VECTOR INDEX `form_10k_chunks` IF NOT EXISTS
    FOR (c:Chunk) ON (c.textEmbedding) 
    OPTIONS { indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'    
    }}
```

### Vector Index Configuration

- **Dimensions (1536)**: Matches OpenAI's text-embedding-ada-002 model output
- **Similarity Function (cosine)**: Optimal for text similarity comparisons
- **Property (textEmbedding)**: Where embedding vectors are stored

### Embedding Generation Process

The system uses Neo4j's built-in AI integration to generate embeddings:

```cypher
MATCH (chunk:Chunk) WHERE chunk.textEmbedding IS NULL
WITH chunk, genai.vector.encode(
    chunk.text, 
    "OpenAI", 
    {
        token: $openAiApiKey, 
        endpoint: $openAiEndpoint
    }) AS vector
CALL db.create.setNodeVectorProperty(chunk, "textEmbedding", vector)
```

### Embedding Generation Workflow

1. **Identifies unprocessed chunks**: Only processes nodes without embeddings
2. **Calls OpenAI API**: Uses genai.vector.encode function
3. **Stores embeddings**: Uses db.create.setNodeVectorProperty for efficient storage
4. **Batch processing**: Handles multiple chunks in single transaction

### Similarity Search Implementation

The `neo4j_vector_search()` function implements semantic search:

1. **Query embedding**: Converts user question to vector using same OpenAI model
2. **Vector similarity**: Uses db.index.vector.queryNodes for efficient search
3. **Returns ranked results**: Provides similarity scores and text content
4. **Configurable results**: top_k parameter controls number of results

---

## RAG Implementation with LangChain

### Vector Store Integration

The system creates a LangChain-compatible vector store:

```python
neo4j_vector_store = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name=VECTOR_INDEX_NAME,
    node_label=VECTOR_NODE_LABEL,
    text_node_properties=[VECTOR_SOURCE_PROPERTY],
    embedding_node_property=VECTOR_EMBEDDING_PROPERTY,
)
```

### RAG Chain Configuration

The `RetrievalQAWithSourcesChain` provides question-answering capabilities:

```python
chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0),  # Deterministic responses
    chain_type="stuff",         # Simple concatenation strategy
    retriever=retriever         # Vector store retriever
)
```

### RAG Workflow Process

1. **Question Processing**: User question converted to embedding
2. **Similarity Search**: Vector index returns relevant chunks
3. **Context Assembly**: Retrieved chunks combined into context
4. **LLM Generation**: ChatOpenAI generates answer using context
5. **Source Attribution**: Chain provides source information

### Response Quality Features

- **Temperature=0**: Ensures consistent, factual responses
- **Source tracking**: Maintains provenance of information
- **Context awareness**: Uses only relevant document chunks
- **Graceful handling**: Acknowledges when information isn't available

---

## Best Practices and Optimization

### Chunk Size Optimization

**Recommended Settings:**
- **Chunk Size**: 2000 characters balances context and processing efficiency
- **Overlap**: 200 characters prevents information loss at boundaries
- **Section Limits**: 20 chunks per section manages processing time while maintaining coverage

### Database Performance

**Index Strategy:**
- Create uniqueness constraints on frequently queried properties
- Vector indexes significantly improve similarity search performance
- Regular schema refresh ensures optimal query planning

### Embedding Efficiency

**Best Practices:**
- Check for existing embeddings before generation (WHERE textEmbedding IS NULL)
- Batch embedding generation for multiple chunks
- Use consistent embedding models throughout the pipeline

### Error Handling

**Robust Implementation:**
- MERGE operations prevent duplicate node creation
- Constraint violations handled gracefully
- API rate limiting consideration for embedding generation

### Scalability Considerations

**For Production Deployment:**
- Implement batch processing for large document sets
- Consider embedding caching strategies
- Monitor Neo4j memory usage with large vector indexes
- Implement proper connection pooling

### Memory Management

**Optimization Strategies:**
- Limit chunk processing per section (demonstrated with [:20] slicing)
- Clear unused variables in long-running processes
- Monitor embedding storage requirements

This manual provides a comprehensive foundation for implementing document-based knowledge graphs using Neo4j, with particular attention to the practical aspects of text processing, graph construction, and semantic search integration.