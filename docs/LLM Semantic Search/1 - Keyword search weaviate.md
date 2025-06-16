# Weaviate Keyword Search Developer Manual

## Introduction

Weaviate is an open-source vector database that combines traditional keyword search with modern vector search capabilities. This manual focuses specifically on implementing keyword search functionality using Weaviate's BM25 algorithm, which is particularly effective for finding documents based on exact term matches and traditional information retrieval patterns.

## What is Weaviate?

Weaviate is a cloud-native, modular, real-time vector database that stores both objects and vectors. It allows you to perform both semantic vector searches and traditional keyword searches on your data. The database is designed to scale horizontally and can integrate with various machine learning models for enhanced search capabilities.

### Key Components

- **Vector Database**: Stores data objects with their associated vector embeddings
- **Graph Structure**: Organizes data in a knowledge graph format
- **Multi-modal Support**: Handles text, images, and other data types
- **RESTful and GraphQL APIs**: Provides flexible query interfaces
- **Modular Architecture**: Supports various vectorization modules (Cohere, OpenAI, etc.)

## Understanding Keyword Search in Weaviate

### BM25 Algorithm

Weaviate implements keyword search using the BM25 (Best Matching 25) algorithm, which is a probabilistic ranking function used by search engines to estimate the relevance of documents to a given search query. BM25 is an improvement over the basic TF-IDF (Term Frequency-Inverse Document Frequency) scoring mechanism.

**How BM25 Works:**
1. **Term Frequency (TF)**: Measures how frequently a term appears in a document
2. **Inverse Document Frequency (IDF)**: Measures how rare or common a term is across the entire collection
3. **Document Length Normalization**: Adjusts for document length to prevent bias toward longer documents
4. **Saturation Function**: Prevents term frequency from dominating the score

### When to Use Keyword Search

Keyword search is most effective when:
- Users search for specific terms, names, or exact phrases
- You need to find documents containing particular technical terms
- The search query contains proper nouns, acronyms, or specialized vocabulary
- You want to implement traditional "search engine-like" functionality
- Exact term matching is more important than semantic similarity

## Setting Up Weaviate Connection

### Authentication and Configuration

Weaviate requires proper authentication and configuration to establish a connection:

```python
import weaviate
import os

# Configure authentication
auth_config = weaviate.auth.AuthApiKey(
    api_key=os.environ['WEAVIATE_API_KEY']
)

# Create client connection
client = weaviate.Client(
    url=os.environ['WEAVIATE_API_URL'],
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ['COHERE_API_KEY'],
    }
)
```

**Key Configuration Elements:**
- **API Key**: Your Weaviate instance authentication key
- **URL**: The endpoint of your Weaviate instance
- **Additional Headers**: Integration keys for external services (like Cohere for embeddings)

### Connection Verification

Always verify your connection before proceeding:

```python
client.is_ready()  # Returns True if connection is successful
```

## Implementing Keyword Search

### Basic Search Structure

A keyword search in Weaviate follows this general pattern:

1. **Define the target class**: Specify which data collection to search
2. **Set properties to retrieve**: Choose which fields to return
3. **Apply BM25 query**: Specify the search terms
4. **Add filters**: Optionally filter results by metadata
5. **Set limits**: Control the number of results returned

### The Search Function Breakdown

```python
def keyword_search(query, results_lang='en', properties=["title","url","text"], num_results=3):
```

**Parameters Explained:**
- `query`: The search string containing keywords to find
- `results_lang`: Language filter to restrict results to specific language content
- `properties`: List of object properties/fields to retrieve in results
- `num_results`: Maximum number of results to return

### Query Construction

The search query is built using Weaviate's query builder pattern:

```python
response = (
    client.query.get("Articles", properties)
    .with_bm25(query=query)
    .with_where(where_filter)
    .with_limit(num_results)
    .do()
)
```

**Query Components:**
- `get("Articles", properties)`: Specifies the data class and fields to retrieve
- `with_bm25(query=query)`: Applies BM25 keyword search algorithm
- `with_where(where_filter)`: Adds metadata filtering conditions
- `with_limit(num_results)`: Restricts the number of returned results
- `do()`: Executes the query

### Filtering Results

Filters help narrow down search results based on metadata:

```python
where_filter = {
    "path": ["lang"],           # Field to filter on
    "operator": "Equal",        # Comparison operator
    "valueString": results_lang # Filter value
}
```

**Available Operators:**
- `Equal`: Exact match
- `NotEqual`: Not equal to
- `GreaterThan`: Greater than (for numbers/dates)
- `LessThan`: Less than (for numbers/dates)
- `Like`: Pattern matching for strings
- `ContainsAny`: Array contains any of the specified values

## Search Results and Response Handling

### Result Structure

Weaviate returns results in a nested JSON structure:

```python
result = response['data']['Get']['Articles']
```

The response hierarchy:
- `data`: Root data container
- `Get`: Indicates a retrieval operation
- `Articles`: The specific class/collection queried
- Array of result objects with requested properties

### Processing Results

Results are typically processed by iterating through the returned array:

```python
for item in result:
    title = item.get('title')
    url = item.get('url')
    text = item.get('text')
    # Process each result item
```

### Available Properties

Common properties you might retrieve:
- `title`: Document or article title
- `text`: Main content text
- `url`: Source URL or identifier
- `views`: Popularity metrics
- `lang`: Language identifier
- `_additional`: Weaviate metadata (requires explicit request)

## Best Practices

### Query Optimization

1. **Use Specific Keywords**: More specific terms generally yield better results
2. **Consider Synonyms**: BM25 looks for exact matches, so include variations
3. **Optimize Property Selection**: Only request properties you actually need
4. **Set Appropriate Limits**: Balance between comprehensiveness and performance

### Language Considerations

When working with multilingual content:
- Always specify language filters when relevant
- Understand that BM25 works differently across languages
- Consider language-specific stemming and tokenization
- Test keyword variations in different languages

### Performance Considerations

1. **Indexing**: Ensure proper indexing on searchable fields
2. **Result Limits**: Use reasonable limits to avoid overwhelming responses
3. **Filter Early**: Apply filters to reduce the search space
4. **Property Selection**: Minimize retrieved properties for better performance

## Common Use Cases

### Document Retrieval
Perfect for finding specific documents, articles, or content pieces based on keyword matches.

### FAQ Systems
Excellent for matching user questions to predefined answers based on keyword overlap.

### Product Search
Useful for e-commerce applications where users search for specific product names or features.

### Research and Knowledge Management
Ideal for finding research papers, documentation, or knowledge base articles.

## Troubleshooting

### Common Issues

1. **No Results Returned**
   - Check if keywords exist in the indexed content
   - Verify language filters aren't too restrictive
   - Ensure the target class contains data

2. **Poor Result Quality**
   - Keywords might be too generic
   - Consider adjusting the number of results
   - Review the quality of indexed content

3. **Connection Issues**
   - Verify API keys and URLs
   - Check network connectivity
   - Ensure Weaviate instance is running

### Debugging Tips

1. **Test Connection**: Always verify `client.is_ready()` first
2. **Examine Raw Results**: Print full response objects to understand structure
3. **Start Simple**: Begin with basic queries before adding complex filters
4. **Check Logs**: Review Weaviate instance logs for error details

## Conclusion

Weaviate's keyword search functionality provides a robust foundation for traditional information retrieval tasks. By understanding the BM25 algorithm, proper query construction, and result handling, developers can implement effective search solutions that complement Weaviate's vector search capabilities.

The key to successful implementation lies in understanding your data structure, choosing appropriate properties, and crafting queries that match your users' search patterns. Combined with Weaviate's scalability and integration capabilities, keyword search becomes a powerful tool in your search arsenal.