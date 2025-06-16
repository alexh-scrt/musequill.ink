# Knowledge Graphs and LLMs Developer Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Knowledge Graphs](#understanding-knowledge-graphs)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Exploring Knowledge Graph Structure](#exploring-knowledge-graph-structure)
5. [Querying Knowledge Graphs with Cypher](#querying-knowledge-graphs-with-cypher)
6. [Integrating LLMs with Knowledge Graphs](#integrating-llms-with-knowledge-graphs)
7. [Advanced Query Patterns](#advanced-query-patterns)
8. [Best Practices and Optimization](#best-practices-and-optimization)
9. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Introduction

Knowledge Graphs (KGs) represent structured information as networks of interconnected entities and relationships. When combined with Large Language Models (LLMs), they enable powerful question-answering systems that can understand natural language queries and convert them into precise graph database queries.

This manual focuses on practical implementation of Knowledge Graph systems integrated with LLMs, using Neo4j as the graph database and LangChain for orchestration.

## Understanding Knowledge Graphs

### Core Concepts

**Nodes (Entities)**: Represent real-world objects, concepts, or entities
- Companies, Managers, Addresses, Forms, Chunks

**Relationships (Edges)**: Define connections between nodes
- `OWNS_STOCK_IN`, `LOCATED_AT`, `FILED`, `SECTION`

**Properties**: Attributes that provide additional information about nodes and relationships
- `managerName`, `companyName`, `city`, `state`, `value`

### Graph Schema Design

A well-designed schema is crucial for effective querying:

```cypher
// Example schema from SEC filings
Manager -[:LOCATED_AT]-> Address
Company -[:LOCATED_AT]-> Address  
Manager -[:OWNS_STOCK_IN]-> Company
Company -[:FILED]-> Form
Form -[:SECTION]-> Chunk
```

## Setting Up the Environment

### Required Dependencies

```python
from dotenv import load_dotenv
import os
import textwrap

# LangChain components
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
```

### Configuration

```python
# Environment setup
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') or 'neo4j'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize graph connection
kg = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD, 
    database=NEO4J_DATABASE
)
```

## Exploring Knowledge Graph Structure

### Schema Inspection

Always start by understanding your graph's structure:

```python
# Refresh and display schema
kg.refresh_schema()
print(textwrap.fill(kg.schema, 60))
```

**Why This Matters**: Understanding the schema helps you:
- Write effective queries
- Identify available relationships
- Plan query optimization strategies

### Data Exploration Patterns

#### 1. Basic Entity Retrieval
```cypher
MATCH (mgr:Manager)-[:LOCATED_AT]->(addr:Address)
RETURN mgr, addr
LIMIT 1
```

#### 2. Full-Text Search
```cypher
CALL db.index.fulltext.queryNodes(
    "fullTextManagerNames", 
    "royal bank"
) YIELD node, score
RETURN node.managerName, score LIMIT 1
```

#### 3. Aggregation Queries
```cypher
MATCH (:Manager)-[:LOCATED_AT]->(address:Address)
RETURN address.state as state, 
       count(address.state) as numManagers
ORDER BY numManagers DESC
LIMIT 10
```

## Querying Knowledge Graphs with Cypher

### Essential Query Patterns

#### Geospatial Queries
For location-based analysis:

```cypher
MATCH (address:Address)
WHERE address.city = "Santa Clara"
MATCH (mgr:Manager)-[:LOCATED_AT]->(managerAddress:Address)
WHERE point.distance(address.location, managerAddress.location) < 10000
RETURN mgr.managerName, mgr.managerAddress
```

**Key Insight**: Geospatial queries enable proximity-based insights, crucial for business intelligence applications.

#### Multi-hop Relationship Traversal
```cypher
MATCH (com:Company)-[:FILED]->(f:Form),
      (f)-[s:SECTION]->(c:Chunk)
WHERE s.f10kItem = "item1"
RETURN c.text
```

#### Complex Filtering and Aggregation
```cypher
MATCH (mgr:Manager)-[:LOCATED_AT]->(address:Address),
      (mgr)-[owns:OWNS_STOCK_IN]->(:Company)
WHERE address.city = "San Francisco"
RETURN mgr.managerName, 
       sum(owns.value) as totalInvestmentValue
ORDER BY totalInvestmentValue DESC
LIMIT 10
```

## Integrating LLMs with Knowledge Graphs

### The Challenge

Converting natural language questions into precise Cypher queries requires:
1. Understanding user intent
2. Mapping concepts to graph entities
3. Generating syntactically correct Cypher
4. Handling edge cases and errors

### Solution Architecture

#### 1. GraphCypherQAChain Setup

```python
cypherChain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=kg,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)
```

**Temperature=0**: Ensures deterministic query generation for consistency.

#### 2. Prompt Engineering Strategy

The key to successful LLM-KG integration is effective prompt design:

```python
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Examples: Here are a few examples of generated Cypher statements for particular questions:

# What investment firms are in San Francisco?
MATCH (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
    WHERE mgrAddress.city = 'San Francisco'
RETURN mgr.managerName

The question is:
{question}"""
```

### Few-Shot Learning Approach

#### Why Few-Shot Learning Works

1. **Pattern Recognition**: LLMs learn query patterns from examples
2. **Schema Awareness**: Examples demonstrate proper use of relationships and properties
3. **Output Format**: Examples establish expected response format

#### Progressive Example Addition

Start with basic examples and gradually add complexity:

**Level 1: Simple Entity Queries**
```cypher
# What investment firms are in San Francisco?
MATCH (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
    WHERE mgrAddress.city = 'San Francisco'
RETURN mgr.managerName
```

**Level 2: Geospatial Queries**
```cypher
# What investment firms are near Santa Clara?
MATCH (address:Address)
    WHERE address.city = "Santa Clara"
MATCH (mgr:Manager)-[:LOCATED_AT]->(managerAddress:Address)
    WHERE point.distance(address.location, managerAddress.location) < 10000
RETURN mgr.managerName, mgr.managerAddress
```

**Level 3: Complex Multi-hop Queries**
```cypher
# What does Palo Alto Networks do?
CALL db.index.fulltext.queryNodes(
    "fullTextCompanyNames", 
    "Palo Alto Networks"
) YIELD node, score
WITH node as com
MATCH (com)-[:FILED]->(f:Form),
      (f)-[s:SECTION]->(c:Chunk)
WHERE s.f10kItem = "item1"
RETURN c.text
```

### Implementation Workflow

#### 1. Query Processing Pipeline

```python
def prettyCypherChain(question: str) -> str:
    response = cypherChain.run(question)
    print(textwrap.fill(response, 60))
```

#### 2. Iterative Prompt Refinement

```python
# Update prompt template
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

# Reinitialize chain with new prompt
cypherChain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=kg,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)
```

**Critical Point**: Always reinitialize the chain after prompt updates to ensure changes take effect.

## Advanced Query Patterns

### Full-Text Search Integration

```cypher
CALL db.index.fulltext.queryNodes(
    "fullTextCompanyNames", 
    "Palo Aalto Networks"  -- Note: handles typos!
) YIELD node, score
WITH node as com
MATCH (com)-[:LOCATED_AT]->(comAddress:Address),
      (mgr:Manager)-[:LOCATED_AT]->(mgrAddress:Address)
WHERE point.distance(comAddress.location, mgrAddress.location) < 10000
RETURN mgr, 
       toInteger(point.distance(comAddress.location, mgrAddress.location) / 1000) as distanceKm
ORDER BY distanceKm ASC
LIMIT 10
```

**Advantages**:
- Fuzzy matching capabilities
- Typo tolerance
- Semantic search integration

### Document Retrieval Integration

Combining structured graph queries with unstructured text:

```cypher
MATCH (com)-[:FILED]->(f:Form),
      (f)-[s:SECTION]->(c:Chunk)
WHERE s.f10kItem = "item1"
RETURN c.text
```

This pattern enables:
- Precise entity identification via graph structure
- Rich context retrieval from documents
- Hybrid structured-unstructured queries

## Best Practices and Optimization

### 1. Schema Design Principles

**Normalize Addresses**: Create separate Address nodes rather than storing addresses as properties
```cypher
// Good
Manager -[:LOCATED_AT]-> Address

// Avoid
Manager {address: "123 Main St, City, State"}
```

**Use Appropriate Indexes**:
- Full-text indexes for name searches
- Property indexes for frequent filtering
- Spatial indexes for location queries

### 2. Query Performance Optimization

**Limit Result Sets**: Always use `LIMIT` in production queries
```cypher
MATCH (mgr:Manager)
RETURN mgr
LIMIT 100  -- Prevents overwhelming responses
```

**Strategic Query Ordering**:
```cypher
// Efficient: Filter early
MATCH (mgr:Manager)-[:LOCATED_AT]->(addr:Address)
WHERE addr.city = 'San Francisco'
RETURN mgr.managerName

// Less efficient: Filter late
MATCH (mgr:Manager)-[:LOCATED_AT]->(addr:Address)
RETURN mgr.managerName
WHERE addr.city = 'San Francisco'
```

### 3. LLM Integration Best Practices

**Clear Instructions**: Be explicit about expected output format
```python
"Do not include any explanations or apologies in your responses."
"Do not include any text except the generated Cypher statement."
```

**Schema Validation**: Always validate generated queries against schema
```python
# Include schema in prompt
Schema:
{schema}
```

**Error Handling**: Implement robust error handling for malformed queries
```python
try:
    result = cypherChain.run(question)
    return result
except Exception as e:
    return f"Query error: {str(e)}"
```

### 4. Prompt Engineering Guidelines

**Progressive Complexity**: Start with simple examples, add complexity gradually
**Domain-Specific Examples**: Use examples from your specific domain
**Edge Case Coverage**: Include examples for edge cases and error conditions

## Troubleshooting Common Issues

### 1. Schema Mismatches

**Problem**: LLM generates queries using non-existent relationships
**Solution**: 
- Regularly refresh schema in prompts
- Validate examples against current schema
- Use verbose mode to debug query generation

### 2. Query Performance Issues

**Problem**: Slow query execution
**Solutions**:
- Add appropriate indexes
- Optimize query structure
- Limit result sets
- Use EXPLAIN to analyze query plans

### 3. LLM Hallucination

**Problem**: LLM generates syntactically correct but semantically wrong queries
**Solutions**:
- Provide more specific examples
- Add validation steps
- Use lower temperature settings
- Implement query result validation

### 4. Complex Query Limitations

**Problem**: LLM struggles with very complex multi-hop queries
**Solutions**:
- Break complex queries into simpler components
- Provide step-by-step examples
- Use query templates for common patterns
- Implement fallback mechanisms

## Conclusion

Successfully integrating LLMs with Knowledge Graphs requires:

1. **Deep Understanding** of your graph schema and data patterns
2. **Careful Prompt Engineering** with relevant examples
3. **Iterative Refinement** of prompts based on real-world queries
4. **Robust Error Handling** and validation
5. **Performance Optimization** for production deployment

The combination of structured graph queries and natural language processing opens powerful possibilities for intuitive data exploration and business intelligence applications. The key is to start simple, test thoroughly, and incrementally add complexity as your system matures.

Remember: The goal is not just to generate correct Cypher queries, but to create a system that reliably translates business questions into actionable insights from your Knowledge Graph.