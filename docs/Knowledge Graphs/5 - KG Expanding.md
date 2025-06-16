# Expanding Knowledge Graphs with Multi-Source Data
## A Complete Developer Manual

### Table of Contents
1. [Overview and Architecture](#overview-and-architecture)
2. [Multi-Source Data Integration Strategy](#multi-source-data-integration-strategy)
3. [Entity Creation and Schema Design](#entity-creation-and-schema-design)
4. [Cross-Source Relationship Patterns](#cross-source-relationship-patterns)
5. [Graph-Enhanced RAG Implementation](#graph-enhanced-rag-implementation)
6. [Performance Analysis and Optimization](#performance-analysis-and-optimization)
7. [Scaling and Best Practices](#scaling-and-best-practices)

---

## Overview and Architecture

### Introduction to Knowledge Graph Expansion

Knowledge graph expansion transforms single-source document repositories into comprehensive information networks by integrating multiple data sources and entity types. This manual analyzes how to expand a document-based knowledge graph (Form 10-K filings) by adding investment data (Form 13 filings) to create a rich financial intelligence system.

### Expansion Architecture Components

The expanded system introduces several new architectural layers:

1. **Multi-Entity Schema**: Company and Manager nodes alongside existing Chunk and Form nodes
2. **Cross-Source Relationships**: OWNS_STOCK_IN and FILED relationships connecting different data types
3. **Data Integration Pipeline**: CSV processing and entity consolidation workflows
4. **Enhanced RAG System**: Graph-traversal-augmented retrieval with investment context

### Data Flow Architecture

```
Form 10-K Documents → Chunks → Forms → Companies ←→ Managers ← Form 13 CSV Data
                                    ↓              ↓
                               FILED Relationships  OWNS_STOCK_IN Relationships
                                    ↓              ↓
                          Enhanced RAG Context Generation
```

---

## Multi-Source Data Integration Strategy

### Understanding Data Source Relationships

The expansion connects two distinct SEC filing types:

**Form 10-K (Annual Reports)**:
- Filed by public companies
- Contains business descriptions, financials, risk factors
- Already processed into Chunk and Form nodes

**Form 13 (Investment Holdings)**:
- Filed by institutional investment managers
- Reports holdings in specific companies
- Provides investment values, share counts, reporting periods

### Common Identifier Strategy

The integration relies on standardized financial identifiers:

- **CUSIP6**: 6-character prefix of Committee on Uniform Securities Identification Procedures code
- **CIK**: Central Index Key assigned by SEC
- **Manager CIK**: Unique identifier for investment management firms

### Data Source Analysis

Form 13 CSV structure contains:
- **Manager Information**: managerCik, managerName, managerAddress
- **Investment Details**: cusip, cusip6, shares, value
- **Company Data**: companyName (as reported by manager)
- **Temporal Context**: reportCalendarOrQuarter

### Integration Benefits

Multi-source integration provides:
- **Entity Enrichment**: Companies gain investment context beyond their own filings
- **Relationship Discovery**: Connections between companies and their institutional investors
- **Temporal Analysis**: Investment changes over reporting periods
- **Cross-Validation**: Company names and identifiers verified across sources

---

## Entity Creation and Schema Design

### Company Entity Schema

Company nodes serve as the central hub connecting document filings with investment data:

```cypher
MERGE (com:Company {cusip6: $cusip6})
  ON CREATE
    SET com.companyName = $companyName,
        com.cusip = $cusip
```

**Company Node Properties**:
- `cusip6`: Primary identifier for cross-source matching
- `companyName`: Display name from Form 13 data
- `cusip`: Full CUSIP identifier
- `names`: Canonical company names from Form 10-K (added later)

### Company Creation Strategy

The implementation demonstrates progressive entity enrichment:

1. **Initial Creation**: Basic company information from Form 13 data
2. **Name Harmonization**: Update with canonical names from Form 10-K
3. **Cross-Reference Validation**: Verify consistency across data sources

### Manager Entity Schema

Manager nodes represent institutional investment firms:

```cypher
MERGE (mgr:Manager {managerCik: $managerParam.managerCik})
  ON CREATE
    SET mgr.managerName = $managerParam.managerName,
        mgr.managerAddress = $managerParam.managerAddress
```

**Manager Node Properties**:
- `managerCik`: Unique SEC identifier for investment managers
- `managerName`: Institutional firm name
- `managerAddress`: Geographic location information

### Entity Creation Patterns

#### MERGE Operation Benefits

The MERGE pattern provides several critical advantages:

- **Idempotency**: Safe to run multiple times without creating duplicates
- **Incremental Updates**: New data can be added without affecting existing entities
- **Data Consolidation**: Multiple records can reference the same entity
- **Error Recovery**: Process continues even if some records already exist

#### Batch Processing Implementation

```python
for form13 in all_form13s:
    kg.query(cypher, params={'managerParam': form13})
```

This approach:
- **Processes all records**: Ensures complete dataset coverage
- **Handles duplicates gracefully**: MERGE prevents duplicate entities
- **Scales linearly**: Processing time grows predictably with dataset size
- **Maintains consistency**: All managers created with same schema

---

## Data Integrity and Indexing

### Constraint Implementation

#### Uniqueness Constraints

```cypher
CREATE CONSTRAINT unique_manager 
  IF NOT EXISTS
  FOR (n:Manager) 
  REQUIRE n.managerCik IS UNIQUE
```

**Constraint Benefits**:
- **Data Integrity**: Database prevents duplicate manager entries
- **Automatic Indexing**: Creates performance index on managerCik
- **Error Prevention**: Write operations fail if uniqueness violated
- **Query Optimization**: Unique constraints enable faster lookups

#### Constraint Strategy Considerations

- **Primary Key Selection**: Choose stable, authoritative identifiers
- **Performance Impact**: Constraints add validation overhead but improve query speed
- **Data Quality**: Constraints catch data issues at write time
- **Migration Planning**: Consider constraint impact when modifying existing data

### Full-Text Search Implementation

```cypher
CREATE FULLTEXT INDEX fullTextManagerNames
  IF NOT EXISTS
  FOR (mgr:Manager) 
  ON EACH [mgr.managerName]
```

#### Full-Text Index Capabilities

**Search Functionality**:
```cypher
CALL db.index.fulltext.queryNodes("fullTextManagerNames", 
    "royal bank") YIELD node, score
RETURN node.managerName, score
```

**Full-Text Advantages**:
- **Fuzzy Matching**: Finds "Royal Bank of Canada" when searching "royal bank"
- **Partial Queries**: Supports incomplete name searches
- **Relevance Scoring**: Results ranked by match quality
- **Multi-Word Search**: Handles complex institutional names

#### Index Performance Considerations

- **Index Size**: Full-text indexes require additional storage
- **Update Overhead**: Index maintenance during data modifications
- **Query Performance**: Significantly faster than LIKE pattern matching
- **Language Support**: Handles various text analysis requirements

---

## Cross-Source Relationship Patterns

### Investment Relationship Schema

The OWNS_STOCK_IN relationship connects managers to companies with rich contextual data:

```cypher
MERGE (mgr)-[owns:OWNS_STOCK_IN { 
    reportCalendarOrQuarter: $ownsParam.reportCalendarOrQuarter
}]->(com)
ON CREATE
    SET owns.value = toFloat($ownsParam.value), 
        owns.shares = toInteger($ownsParam.shares)
```

#### Relationship Property Design

**Temporal Properties**:
- `reportCalendarOrQuarter`: Identifies reporting period
- Creates composite key with manager-company pair

**Investment Properties**:
- `value`: Dollar value of holdings (stored as float)
- `shares`: Number of shares held (stored as integer)

**Data Type Conversion Benefits**:
- **Numerical Operations**: Enables mathematical calculations
- **Sorting Capability**: ORDER BY value/shares for ranking
- **Aggregation Support**: SUM, AVG operations on holdings
- **Storage Efficiency**: Optimized storage for numerical data

### Document Ownership Relationships

#### FILED Relationship Creation

```cypher
MATCH (com:Company), (form:Form)
  WHERE com.cusip6 = form.cusip6
MERGE (com)-[:FILED]->(form)
```

**FILED Relationship Benefits**:
- **Document Attribution**: Clear ownership of SEC filings
- **Traversal Enabling**: Navigate from companies to their disclosures
- **Data Lineage**: Track information sources and provenance
- **Contextual Queries**: Find documents related to specific companies

#### Cross-Source Validation

The FILED relationship enables data validation across sources:
- **Identifier Consistency**: Verify CUSIP6 matches between Form 10-K and Form 13
- **Name Harmonization**: Update company names with authoritative Form 10-K data
- **Completeness Checking**: Identify companies with missing document filings

### Composite Relationship Keys

#### Multi-Property Uniqueness

The investment relationship uses composite keys combining:
- Source Manager (managerCik)
- Target Company (cusip6)  
- Reporting Period (reportCalendarOrQuarter)

**Composite Key Advantages**:
- **Temporal Tracking**: Same manager-company pair across multiple quarters  
- **Historical Analysis**: Track investment changes over time
- **Update Safety**: Prevents accidental overwriting of historical data
- **Analytical Flexibility**: Enables time-series investment analysis

---

## Graph-Enhanced RAG Implementation

### Investment Context Generation

#### Dynamic Context Assembly

The implementation demonstrates sophisticated context generation through graph traversal:

```cypher
MATCH (:Chunk {chunkId: $chunkIdParam})-[:PART_OF]->(f:Form),
      (com:Company)-[:FILED]->(f),
      (mgr:Manager)-[owns:OWNS_STOCK_IN]->(com)
RETURN mgr.managerName + " owns " + owns.shares + 
       " shares of " + com.companyName + 
       " at a value of $" + 
       apoc.number.format(toInteger(owns.value)) AS text
```

#### Context Generation Benefits

**Human-Readable Output**:
- **Structured Sentences**: Natural language investment descriptions
- **Formatted Numbers**: Proper currency and share formatting using APOC functions
- **Contextual Relevance**: Information directly related to document content

**Dynamic Assembly Process**:
1. **Start from Document Chunk**: Begin with user's query target
2. **Traverse to Company**: Follow document ownership path
3. **Find Investors**: Identify managers with holdings
4. **Format Information**: Create readable investment summaries

### Advanced Retrieval Query Architecture

#### Multi-Source Information Integration

```cypher
MATCH (node)-[:PART_OF]->(f:Form),
      (f)<-[:FILED]-(com:Company),
      (com)<-[owns:OWNS_STOCK_IN]-(mgr:Manager)
WITH node, score, mgr, owns, com 
    ORDER BY owns.shares DESC LIMIT 10
WITH collect (
    mgr.managerName + 
    " owns " + owns.shares + 
    " shares in " + com.companyName + 
    " at a value of $" + 
    apoc.number.format(toInteger(owns.value)) + "." 
) AS investment_statements, node, score
RETURN apoc.text.join(investment_statements, "\n") + 
    "\n" + node.text AS text
```

#### Advanced Query Pattern Analysis

**Multi-Stage Processing**:
1. **Graph Traversal**: Navigate from document chunks to investment data
2. **Data Aggregation**: Collect and rank investment information
3. **Context Assembly**: Combine investment context with document text
4. **Result Formatting**: Create coherent, readable output

**Ranking and Limitation**:
- **ORDER BY owns.shares DESC**: Prioritize largest holdings
- **LIMIT 10**: Control context size for LLM processing
- **Relevance Focus**: Most significant investments first

#### Context Enhancement Strategy

**Information Layering**:
- **Primary Content**: Original document text remains intact
- **Contextual Augmentation**: Investment information prepended to document content
- **Seamless Integration**: Combined context appears as unified text to LLM

**Dynamic Relevance**:
- **Query-Dependent**: Investment context only added when relevant
- **Adaptive Sizing**: Context amount varies based on available data
- **Performance Balanced**: Context generation limited to prevent excessive processing

---

## Performance Analysis and Optimization

### Comparative RAG Performance

#### Configuration Comparison

The implementation demonstrates two distinct RAG configurations:

**Plain Retrieval Chain**:
```python
plain_chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever
)
```

**Graph-Enhanced Retrieval Chain**:
```python
investment_chain = RetrievalQAWithSourcesChain.from_chain_type(
    ChatOpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever_with_investments
)
```

#### Performance Analysis Results

**Question Dependency**:
- **Generic Queries**: "Tell me about NetApp" shows minimal investment context utilization
- **Specific Queries**: "Tell me about NetApp investors" fully leverages graph-enhanced context

**Context Utilization Patterns**:
- **Relevance Matching**: LLM selectively uses context based on query intent
- **Information Integration**: Graph context seamlessly integrated with document content
- **Factual Grounding**: Investment data provides concrete, verifiable information

### Query Performance Characteristics

#### Graph Traversal Efficiency

**Multi-Hop Query Performance**:
- **Index Utilization**: Constraints and indexes optimize traversal paths
- **Relationship Direction**: Queries follow natural relationship directions
- **Result Limiting**: LIMIT clauses prevent excessive result sets

**Memory Usage Patterns**:
- **Batch Processing**: Large dataset processing managed through iterations
- **Context Size Control**: Limited investment statements prevent memory issues
- **Query Optimization**: Efficient Cypher patterns minimize resource usage

#### Scaling Performance Considerations

**Data Volume Impact**:
- **Linear Growth**: Processing time scales predictably with data size
- **Index Effectiveness**: Performance remains stable with proper indexing
- **Memory Management**: Batch operations prevent memory overflow

**Concurrent Access**:
- **Connection Pooling**: Multiple simultaneous queries supported
- **Query Isolation**: Individual queries don't interfere with each other
- **Resource Sharing**: Database efficiently manages concurrent operations

---

## Scaling and Best Practices

### Data Integration Scaling

#### Batch Processing Strategies

**Entity Creation Order**:
1. **Create Nodes First**: Ensure all entities exist before relationships
2. **Validate Identifiers**: Check for required fields before processing
3. **Progress Tracking**: Monitor processing status for large datasets
4. **Error Handling**: Continue processing despite individual record failures

**Transaction Management**:
- **Batch Sizes**: Optimize transaction size for memory and performance
- **Commit Strategies**: Balance consistency with processing efficiency
- **Rollback Planning**: Handle partial failures gracefully

#### Schema Evolution Planning

**Flexible Design Principles**:
- **Consistent Naming**: Use standard conventions across entity types
- **Property Extensibility**: Design for additional properties over time
- **Relationship Versioning**: Plan for relationship schema changes
- **Migration Strategies**: Design backward-compatible updates

**Data Quality Management**:
- **Validation Rules**: Implement checks for data consistency
- **Constraint Strategy**: Balance integrity with flexibility
- **Monitoring Systems**: Track data quality metrics over time

### Performance Optimization Strategies

#### Index Strategy Scaling

**Primary Index Types**:
- **Uniqueness Constraints**: All primary keys should have unique constraints
- **Range Indexes**: Numerical properties benefit from range indexing
- **Full-Text Indexes**: Text search fields require specialized indexing
- **Composite Indexes**: Multi-property queries benefit from composite indexes

**Index Maintenance**:
- **Performance Monitoring**: Track query execution times
- **Index Usage Analysis**: Identify unused or ineffective indexes
- **Maintenance Scheduling**: Plan for index rebuilding and optimization

#### Query Optimization Techniques

**Cypher Best Practices**:
- **EXPLAIN Analysis**: Use EXPLAIN to understand query execution plans
- **Filter Early**: Apply WHERE clauses as early as possible
- **Limit Results**: Use LIMIT to prevent excessive result sets
- **Index Hints**: Guide query planner with index hints when necessary

**Memory Management**:
- **Connection Pooling**: Reuse database connections efficiently
- **Query Caching**: Cache frequently executed queries
- **Resource Monitoring**: Track memory usage and garbage collection

### Knowledge Graph Expansion Best Practices

#### Data Integration Principles

1. **Common Identifiers**: Standardize on authoritative identifiers (CUSIP, CIK)
2. **Data Quality First**: Implement validation before loading
3. **Incremental Loading**: Design for ongoing updates and additions
4. **Source Attribution**: Maintain clear data provenance

#### Entity Relationship Design

1. **Semantic Clarity**: Relationship names should be self-explanatory
2. **Property Completeness**: Include all relevant contextual information
3. **Temporal Awareness**: Handle time-varying data appropriately
4. **Cardinality Planning**: Design for expected relationship patterns

#### RAG Enhancement Strategies

1. **Context Relevance**: Generate context that matches query intent
2. **Information Ranking**: Order context by importance or recency
3. **Format Consistency**: Maintain readable, structured output
4. **Performance Balance**: Balance context richness with response time

#### Maintenance and Evolution

1. **Schema Documentation**: Maintain clear entity and relationship documentation
2. **Data Validation**: Implement regular data quality checks
3. **Performance Monitoring**: Track system performance as data grows
4. **Evolution Planning**: Design for future data sources and requirements

---

## Conclusion

### Key Expansion Insights

Knowledge graph expansion transforms document repositories into comprehensive information networks by:

1. **Multi-Source Integration**: Connecting heterogeneous data through common identifiers
2. **Entity Enrichment**: Adding business context beyond document content
3. **Relationship Discovery**: Revealing connections between entities across data sources
4. **Enhanced Analytics**: Enabling complex queries spanning multiple entity types

### Implementation Success Factors

1. **Data Quality Management**: Robust constraints and validation ensure reliable expansion
2. **Performance Optimization**: Proper indexing and query design maintain system responsiveness  
3. **Schema Flexibility**: Extensible design accommodates future data sources
4. **RAG Enhancement**: Graph-aware context generation significantly improves answer quality

### Future Expansion Opportunities

The patterns demonstrated in this manual can be applied to:
- **Additional SEC Filings**: Form 8-K, proxy statements, insider trading reports
- **Market Data Integration**: Stock prices, analyst ratings, financial metrics
- **News and Events**: Press releases, earnings calls, regulatory actions
- **Geographic Data**: Company locations, market presence, regulatory jurisdictions

This comprehensive approach to knowledge graph expansion provides a foundation for building sophisticated information systems that connect documents, entities, and relationships across multiple data domains, enabling advanced analytics and enhanced AI-powered applications.