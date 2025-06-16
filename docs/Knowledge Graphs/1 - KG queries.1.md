# Knowledge Graphs Developer Manual
## A Complete Guide to Understanding, Querying, and Leveraging Knowledge Graphs with Neo4j

---

## Table of Contents

**Part I: Foundations**
1. [Introduction to Knowledge Graphs](#introduction-to-knowledge-graphs)
2. [Understanding the Graph Data Model](#understanding-the-graph-data-model)
3. [Setting Up Neo4j Environment](#setting-up-neo4j-environment)

**Part II: Querying with Cypher**
4. [Cypher Query Language Fundamentals](#cypher-query-language-fundamentals)
5. [Basic Pattern Matching](#basic-pattern-matching)
6. [Advanced Query Techniques](#advanced-query-techniques)
7. [Data Manipulation Operations](#data-manipulation-operations)

**Part III: Vector Embeddings and Modern RAG**
8. [Introduction to Vector Embeddings in Graphs](#introduction-to-vector-embeddings-in-graphs)
9. [Creating and Populating Vector Indexes](#creating-and-populating-vector-indexes)
10. [Similarity Search and Retrieval](#similarity-search-and-retrieval)

---

## Part I: Foundations

### Introduction to Knowledge Graphs

Knowledge Graphs represent a paradigm shift in how we model, store, and query interconnected data. Unlike traditional relational databases that store data in rigid tables, knowledge graphs use a flexible graph structure consisting of **nodes** (entities) and **relationships** (connections) that mirror how information naturally relates in the real world.

#### What Makes Knowledge Graphs Powerful?

1. **Intuitive Data Modeling**: Graph structures naturally represent relationships between entities
2. **Flexible Schema**: Easy to add new types of entities and relationships without restructuring
3. **Efficient Traversal**: Optimized for finding connections and paths between related data
4. **Semantic Understanding**: Can represent complex, multi-hop relationships and dependencies

#### Real-World Applications

Knowledge graphs power many systems you interact with daily:
- **Search engines** use them to understand entity relationships and provide contextual results
- **Recommendation systems** leverage graph connections to suggest relevant content
- **Fraud detection** systems identify suspicious patterns through relationship analysis
- **AI and machine learning** models use graph structure for enhanced context understanding

### Understanding the Graph Data Model

The examples in our notebooks use a movie database that perfectly illustrates graph concepts:

#### Core Components

**Nodes (Entities)**:
- `Person`: Represents actors, directors, producers
- `Movie`: Represents films with properties like title, release year, tagline

**Relationships (Edges)**:
- `ACTED_IN`: Connects Person nodes to Movie nodes
- `DIRECTED`: Links directors to their films
- `WORKS_WITH`: Professional relationships between people

**Properties**:
- Nodes and relationships can have key-value properties
- Examples: `name`, `title`, `released`, `tagline`

#### Graph vs. Relational Thinking

In a relational database, you might have separate tables for movies, people, and their relationships, requiring complex JOINs to find connections. In a graph database, these relationships are first-class citizens, making traversal queries natural and efficient.

### Setting Up Neo4j Environment

The notebooks demonstrate connection setup using Python and LangChain:

```python
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE')

# Initialize connection
kg = Neo4jGraph(
    url=NEO4J_URI, 
    username=NEO4J_USERNAME, 
    password=NEO4J_PASSWORD, 
    database=NEO4J_DATABASE
)
```

**Key Configuration Elements**:
- **URI**: Connection endpoint to your Neo4j instance
- **Credentials**: Username and password for authentication
- **Database**: Specific database name within Neo4j instance
- **LangChain Integration**: Provides Python wrapper for executing Cypher queries

---

## Part II: Querying with Cypher

### Cypher Query Language Fundamentals

Cypher is Neo4j's declarative query language, designed to be intuitive and readable. It uses ASCII art-like syntax to represent graph patterns.

#### Basic Syntax Patterns

**Node Pattern**: `(variable:Label {property: value})`
- Parentheses represent nodes
- Variables (like `m`, `person`) reference nodes in queries
- Labels (like `:Movie`, `:Person`) categorize node types
- Properties filter or specify node characteristics

**Relationship Pattern**: `-[:RELATIONSHIP_TYPE]->`
- Hyphens and arrows show relationship direction
- Square brackets contain relationship types
- Direction matters: `->` vs `<-` vs `-` (bidirectional)

### Basic Pattern Matching

#### Counting All Nodes

The simplest query demonstrates basic node matching:

```cypher
MATCH (n) 
RETURN count(n)
```

**What this does**:
- `MATCH (n)`: Finds all nodes in the database
- `RETURN count(n)`: Returns the total count
- Variable `n` is arbitrary - could be any name

#### Filtering by Labels

More specific queries use node labels:

```cypher
MATCH (m:Movie) 
RETURN count(m) AS numberOfMovies
```

**Key concepts**:
- `:Movie` label restricts matching to Movie nodes only
- `AS numberOfMovies` creates an alias for the result
- More readable variable names (`m` instead of `n`) improve query clarity

#### Property-Based Filtering

Finding specific entities by their properties:

```cypher
MATCH (tom:Person {name:"Tom Hanks"}) 
RETURN tom
```

**Analysis**:
- Combines label filtering (`:Person`) with property matching (`{name:"Tom Hanks"}`)
- Returns the entire node object with all its properties
- Efficient when you know specific property values

#### Selective Property Retrieval

Instead of returning entire nodes, extract specific properties:

```cypher
MATCH (cloudAtlas:Movie {title:"Cloud Atlas"}) 
RETURN cloudAtlas.released, cloudAtlas.tagline
```

**Benefits**:
- Reduces data transfer by selecting only needed properties
- Cleaner result format for specific use cases
- Dot notation (`node.property`) accesses individual properties

### Advanced Query Techniques

#### Conditional Filtering with WHERE

The `WHERE` clause enables complex filtering logic:

```cypher
MATCH (nineties:Movie) 
WHERE nineties.released >= 1990 
  AND nineties.released < 2000 
RETURN nineties.title
```

**Advanced filtering capabilities**:
- Numerical comparisons (`>=`, `<`, `=`)
- Logical operators (`AND`, `OR`, `NOT`)
- String pattern matching (not shown but available)
- Null checking and property existence tests

#### Multi-Node Pattern Matching

The real power of graph queries emerges with relationship traversal:

```cypher
MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie) 
RETURN actor.name, movie.title LIMIT 10
```

**Pattern breakdown**:
- Two nodes: `(actor:Person)` and `(movie:Movie)`
- Relationship: `-[:ACTED_IN]->` shows directed connection
- Result: All actor-movie pairs where the actor appeared in the movie
- `LIMIT 10` controls result size for performance

#### Complex Relationship Patterns

Finding co-actors demonstrates advanced pattern matching:

```cypher
MATCH (tom:Person {name:"Tom Hanks"})-[:ACTED_IN]->(m)<-[:ACTED_IN]-(coActors) 
RETURN coActors.name, m.title
```

**Pattern explanation**:
- Start with Tom Hanks: `(tom:Person {name:"Tom Hanks"})`
- Find his movies: `-[:ACTED_IN]->(m)`
- Find other actors in same movies: `<-[:ACTED_IN]-(coActors)`
- The `m` variable represents the connecting movie
- Results show all actors who appeared in movies with Tom Hanks

### Data Manipulation Operations

#### Deleting Relationships

Remove specific connections while preserving nodes:

```cypher
MATCH (emil:Person {name:"Emil Eifrem"})-[actedIn:ACTED_IN]->(movie:Movie)
DELETE actedIn
```

**Important concepts**:
- Square brackets capture the relationship: `[actedIn:ACTED_IN]`
- `DELETE` removes only the relationship, not the connected nodes
- Nodes remain intact unless explicitly deleted
- This is safer than deleting nodes, which would remove all connected relationships

#### Creating New Nodes

Add entities to the graph:

```cypher
CREATE (andreas:Person {name:"Andreas"})
RETURN andreas
```

**Creation details**:
- `CREATE` always creates new nodes (even if similar ones exist)
- Properties are set during creation: `{name:"Andreas"}`
- `RETURN` confirms successful creation and shows the new node

#### Merging Relationships

Create relationships between existing nodes:

```cypher
MATCH (andreas:Person {name:"Andreas"}), (emil:Person {name:"Emil Eifrem"})
MERGE (andreas)-[hasRelationship:WORKS_WITH]->(emil)
RETURN andreas, hasRelationship, emil
```

**MERGE vs CREATE**:
- `MERGE` creates the relationship only if it doesn't already exist
- Prevents duplicate relationships
- `MATCH` first finds both existing nodes
- Result returns all three components: both nodes and the relationship

---

## Part III: Vector Embeddings and Modern RAG

### Introduction to Vector Embeddings in Graphs

Modern knowledge graphs extend beyond traditional property-based queries by incorporating vector embeddings. This enables semantic similarity search, where queries match meaning rather than exact text.

#### Why Vector Embeddings in Graphs?

Traditional graph queries excel at exact matches and structural patterns, but struggle with:
- Semantic similarity ("love" vs "romance" vs "affection")
- Fuzzy matching for user queries
- Content-based recommendations
- Integration with AI/ML workflows

Vector embeddings solve these challenges by:
- Converting text to numerical representations
- Enabling mathematical similarity calculations
- Supporting natural language queries
- Bridging graph data with AI models

### Creating and Populating Vector Indexes

#### Vector Index Definition

The notebook demonstrates creating a vector index for movie taglines:

```cypher
CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS
FOR (m:Movie) ON (m.taglineEmbedding) 
OPTIONS { indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
```

**Configuration breakdown**:
- **Index name**: `movie_tagline_embeddings` for reference
- **Node pattern**: `FOR (m:Movie)` specifies which nodes to index
- **Property**: `ON (m.taglineEmbedding)` indicates which property contains vectors
- **Dimensions**: `1536` matches OpenAI's embedding size
- **Similarity function**: `cosine` for measuring vector similarity

#### Embedding Generation and Storage

The process of populating embeddings involves external API calls:

```cypher
MATCH (movie:Movie) WHERE movie.tagline IS NOT NULL
WITH movie, genai.vector.encode(
    movie.tagline, 
    "OpenAI", 
    {
      token: $openAiApiKey,
      endpoint: $openAiEndpoint
    }) AS vector
CALL db.create.setNodeVectorProperty(movie, "taglineEmbedding", vector)
```

**Process flow**:
1. **Filter nodes**: Only movies with existing taglines
2. **Generate embeddings**: `genai.vector.encode()` calls OpenAI API
3. **Store vectors**: `setNodeVectorProperty()` adds embedding to node
4. **Parameterization**: API credentials passed securely as parameters

**Why this approach works**:
- Embeddings capture semantic meaning of taglines
- 1536-dimensional vectors represent complex textual concepts
- Cosine similarity measures semantic closeness
- Integration with external AI services (OpenAI) provides state-of-the-art embeddings

### Similarity Search and Retrieval

#### Semantic Query Processing

The power of vector embeddings becomes apparent in similarity search:

```cypher
WITH genai.vector.encode(
    $question, 
    "OpenAI", 
    {
      token: $openAiApiKey,
      endpoint: $openAiEndpoint
    }) AS question_embedding
CALL db.index.vector.queryNodes(
    'movie_tagline_embeddings', 
    $top_k, 
    question_embedding
    ) YIELD node AS movie, score
RETURN movie.title, movie.tagline, score
```

**Query execution flow**:
1. **Encode question**: Convert user question to vector using same method as stored embeddings
2. **Vector search**: `db.index.vector.queryNodes()` finds most similar vectors
3. **Ranking**: Results ordered by similarity score (higher = more similar)
4. **Retrieval**: Returns relevant movies with their taglines and similarity scores

#### Example Results Analysis

For the question "What movies are about love?", the system returns:

1. **"Joe Versus the Volcano"** - "A story of love, lava and burning desire." (Score: 0.906)
2. **"Snow Falling on Cedars"** - "First loves last. Forever." (Score: 0.901)
3. **"Sleepless in Seattle"** - "What if someone you never met... was the only someone for you?" (Score: 0.895)

**Why this works**:
- Direct mention of "love" in taglines scores highest
- Semantic concepts like "first loves" and romantic scenarios score well
- Mathematical similarity captures meaning beyond exact word matches
- Scores provide confidence levels for ranking results

#### Practical Applications

This vector similarity approach enables:

**Content Discovery**:
- Users can ask natural language questions
- System finds semantically relevant content
- No need for exact keyword matching

**Recommendation Systems**:
- Find similar content based on descriptions
- Cross-reference user preferences with content semantics
- Enable exploratory browsing

**RAG (Retrieval Augmented Generation)**:
- Retrieve relevant context for AI model queries
- Combine structured graph data with semantic search
- Support conversational AI with knowledge grounding

---

## Summary and Best Practices

### Key Takeaways

**Graph Modeling Advantages**:
- Natural representation of connected data
- Flexible schema evolution
- Efficient relationship traversal
- Intuitive query patterns

**Cypher Query Patterns**:
- Start simple with basic node matching
- Build complexity through relationship patterns
- Use WHERE clauses for advanced filtering
- Combine pattern matching with property filtering

**Vector Integration Benefits**:
- Semantic search capabilities
- Natural language query support
- AI/ML workflow integration
- Enhanced content discovery

### Development Best Practices

1. **Start with Clear Data Models**: Define your nodes, relationships, and properties before implementation
2. **Use Descriptive Variables**: Make Cypher queries readable with meaningful variable names
3. **Index Strategically**: Create indexes for frequently queried properties and vector searches
4. **Parameterize Queries**: Use parameters for dynamic values and security
5. **Monitor Performance**: Use `LIMIT` clauses and profiling for query optimization
6. **Combine Traditional and Vector Queries**: Leverage both exact matching and semantic similarity as needed

This comprehensive approach to knowledge graphs provides a foundation for building intelligent, connected data systems that bridge traditional database capabilities with modern AI-powered search and discovery.