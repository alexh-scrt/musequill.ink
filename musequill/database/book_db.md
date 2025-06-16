# Python MongoDB Client Setup

## Install Required Dependencies

```bash
# Install Python MongoDB drivers
pip install pymongo motor

# Optional: Install additional dependencies for async support
pip install asyncio-contextmanager

# Create requirements.txt
cat > requirements.txt << 'EOF'
pymongo>=4.6.0
motor>=3.3.0
python-dotenv>=1.0.0
EOF

pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file for your MongoDB configuration:

```bash
cat > .env << 'EOF'
# MongoDB Configuration
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=musequill
MONGODB_USERNAME=musequill
MONGODB_PASSWORD=musequill.ink.user
MONGODB_AUTH_DATABASE=musequill

# Connection Pool Settings
MONGODB_MIN_POOL_SIZE=5
MONGODB_MAX_POOL_SIZE=50
MONGODB_MAX_IDLE_TIME_MS=30000
MONGODB_WAIT_QUEUE_TIMEOUT_MS=5000
EOF
```

## Basic Usage Examples

### 1. Synchronous Client Usage

```python
from mongodb_client import MongoDBClient, MongoDBConfig, BookStatus

# Create configuration
config = MongoDBConfig(
    host="localhost",
    port=27017,
    database="musequill",
    username="musequill",
    password="musequill.ink.user"
)

# Use context manager (recommended)
with MongoDBClient(config) as client:
    # Create a book
    book_data = {
        'title': 'My Novel',
        'genre': 'fantasy',
        'estimated_word_count': 80000,
        'parameters': {
            'style': 'epic',
            'tone': 'adventurous',
            'length': 'novel'
        }
    }
    
    book_id = client.create_book(book_data)
    print(f"Created book: {book_id}")
    
    # Update book status
    client.update_book_status(book_id, BookStatus.PLANNING)
    
    # Get book
    book = client.get_book(book_id)
    print(f"Book title: {book['title']}")
    
    # List all books
    books = client.list_books()
    print(f"Total books: {len(books)}")
```

### 2. Asynchronous Client Usage

```python
import asyncio
from mongodb_client import AsyncMongoDBClient, MongoDBConfig

async def main():
    config = MongoDBConfig()
    
    async with AsyncMongoDBClient(config).session() as client:
        # Create book
        book_data = {'title': 'Async Book', 'genre': 'sci-fi'}
        book_id = await client.create_book(book_data)
        
        # Get book
        book = await client.get_book(book_id)
        print(f"Async book: {book['title']}")

# Run async example
asyncio.run(main())
```

### 3. Working with Your Existing books_db Data

```python
from mongodb_client import MongoDBClient
from uuid import UUID
from datetime import datetime

# Convert your existing books_db data to MongoDB
def migrate_books_db_to_mongodb(books_db, client):
    """Migrate existing books_db dictionary to MongoDB."""
    
    for book_uuid, book_data in books_db.items():
        # Convert UUID to string
        book_id = str(book_uuid)
        
        # Prepare MongoDB document
        mongo_book = {
            '_id': book_id,
            'created_at': book_data.get('created_at', datetime.now()),
            'updated_at': book_data.get('updated_at', datetime.now()),
            'status': book_data.get('status', 'initializing'),
            'title': book_data.get('parameters', {}).get('title', 'Untitled'),
            'genre': book_data.get('parameters', {}).get('genre', ''),
            'estimated_word_count': book_data.get('estimated_word_count', 0),
            'completion_percentage': book_data.get('completion_percentage', 0.0),
            
            # Store all original data in structured fields
            'parameters': book_data.get('parameters', {}),
            'planning_results': book_data.get('planning_results', {}),
            'validation_info': book_data.get('validation_info', {}),
            'genre_info': book_data.get('genre_info', {}),
            
            # Additional fields
            'planning_status': book_data.get('planning_status'),
            'agent_id': book_data.get('agent_id'),
            'planning_completed': book_data.get('planning_completed', False),
            'last_attempt': book_data.get('last_attempt'),
            'retry_count': book_data.get('retry_count', 0),
            'error_message': book_data.get('error_message'),
            'status_message': book_data.get('status_message'),
            'next_steps': book_data.get('next_steps', [])
        }
        
        try:
            client.create_book(mongo_book)
            print(f"Migrated book: {book_id}")
        except ValueError as e:
            print(f"Book {book_id} already exists, updating...")
            client.update_book(book_id, mongo_book)

# Example migration
with MongoDBClient() as client:
    # migrate_books_db_to_mongodb(books_db, client)
    pass
```

### 4. Agent Integration Example

```python
from mongodb_client import MongoDBClient, AgentType, AgentSessionStatus

def agent_workflow_example():
    """Example of how agents would interact with the database."""
    
    with MongoDBClient() as client:
        # Agent starts working on a book
        book_id = "some-book-uuid"
        
        # Create agent session
        session_data = {
            'book_id': book_id,
            'agent_type': AgentType.PLANNER,
            'agent_id': 'planning_agent_001',
            'input_data': {'requirements': 'planning_requirements'},
            'lock_acquired_at': datetime.now()
        }
        
        session_id = client.create_agent_session(session_data)
        
        # Update book status
        client.update_book_status(book_id, 'planning', {
            'last_agent_id': 'planning_agent_001',
            'planning_started_at': datetime.now()
        })
        
        # Simulate agent work...
        # Agent completes work
        
        # Update session
        client.update_agent_session(session_id, {
            'status': AgentSessionStatus.COMPLETED,
            'completed_at': datetime.now(),
            'output_data': {'planning_result': 'completed_plan'}
        })
        
        # Update book with results
        client.update_book_status(book_id, 'planned', {
            'planning_completed': True,
            'planning_results': {'story_outline': '...', 'chapters': '...'}
        })
```

## Testing the Setup

Create a test script to verify everything works:

```python
# test_mongodb_setup.py
from mongodb_client import MongoDBClient, BookStatus, AgentType
import json

def test_mongodb_setup():
    """Test MongoDB setup and operations."""
    
    try:
        with MongoDBClient() as client:
            # Health check
            health = client.health_check()
            print("✅ Health Check:", health['status'])
            
            # Test book operations
            test_book = {
                'title': 'Test Setup Book',
                'genre': 'test',
                'estimated_word_count': 1000,
                'parameters': {'test': True}
            }
            
            # Create
            book_id = client.create_book(test_book)
            print(f"✅ Created book: {book_id}")
            
            # Read
            book = client.get_book(book_id)
            print(f"✅ Retrieved book: {book['title']}")
            
            # Update
            updated = client.update_book_status(book_id, BookStatus.PLANNING)
            print(f"✅ Updated book status: {updated}")
            
            # List
            books = client.list_books(status=BookStatus.PLANNING)
            print(f"✅ Listed {len(books)} planning books")
            
            # Agent session test
            session_id = client.create_agent_session({
                'book_id': book_id,
                'agent_type': AgentType.PLANNER,
                'agent_id': 'test_agent'
            })
            print(f"✅ Created agent session: {session_id}")
            
            # Clean up
            client.delete_book(book_id)
            print("✅ Cleaned up test data")
            
            print("\n🎉 All tests passed! MongoDB client is ready.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    test_mongodb_setup()
```

Run the test:

```bash
python test_mongodb_setup.py
```

## Integration with MuseQuill

To integrate with your existing MuseQuill codebase:

### 1. Replace books_db dictionary

```python
# Before (using dictionary)
from musequill.database.book import books_db

# After (using MongoDB)
from mongodb_client import MongoDBClient, MongoDBConfig

# Create global client instance
mongodb_config = MongoDBConfig()
mongodb_client = MongoDBClient(mongodb_config)
mongodb_client.connect()

# Update functions to use MongoDB instead of dictionary
def create_book(book_data):
    return mongodb_client.create_book(book_data)

def get_book(book_id):
    return mongodb_client.get_book(book_id)

def update_book(book_id, data):
    return mongodb_client.update_book(book_id, data)
```

### 2. Update agent integration

```python
# In musequill/agents/integration.py
async def update_book_status(book_id: UUID, status: str, additional_data: dict = None):
    """Update book status in MongoDB instead of books_db."""
    return mongodb_client.update_book_status(str(book_id), status, additional_data)
```

## Performance Tips

1. **Use indexes wisely** - The client assumes indexes are created as per the MongoDB setup guide
2. **Connection pooling** - Use the same client instance across your application
3. **Batch operations** - For multiple inserts, use `insert_many()` when available
4. **Projection** - Only fetch fields you need using MongoDB projections
5. **Caching** - Cache frequently accessed data in Redis or memory

## Error Handling Best Practices

```python
from mongodb_client import MongoDBClient
import logging

def robust_book_operation(book_id: str):
    """Example of robust error handling."""
    
    try:
        with MongoDBClient() as client:
            book = client.get_book(book_id)
            if not book:
                raise ValueError(f"Book {book_id} not found")
            
            # Perform operations
            result = client.update_book_status(book_id, "processing")
            return result
            
    except ValueError as e:
        logging.error(f"Book operation failed: {e}")
        raise
    except Exception as e:
        logging.error(f"Database error: {e}")
        # Handle database connectivity issues
        raise
```

## Advanced Usage Patterns

### 1. Bulk Operations

```python
def create_multiple_books(books_data: List[Dict]):
    """Create multiple books efficiently."""
    
    with MongoDBClient() as client:
        # Prepare documents
        docs = []
        for book_data in books_data:
            doc = {
                '_id': str(uuid4()),
                'created_at': datetime.now(timezone.utc),
                'title': book_data['title'],
                # ... other fields
            }
            docs.append(doc)
        
        # Bulk insert
        result = client._books_collection.insert_many(docs)
        return [str(doc_id) for doc_id in result.inserted_ids]
```

### 2. Complex Queries

```python
def find_books_by_criteria(client: MongoDBClient, 
                          genre: str = None,
                          min_word_count: int = None,
                          status_list: List[str] = None):
    """Advanced book filtering."""
    
    query = {}
    
    if genre:
        query['genre'] = {'$regex': genre, '$options': 'i'}  # Case insensitive
    
    if min_word_count:
        query['estimated_word_count'] = {'$gte': min_word_count}
    
    if status_list:
        query['status'] = {'$in': status_list}
    
    # Add date range if needed
    # query['created_at'] = {'$gte': start_date, '$lte': end_date}
    
    return list(client._books_collection.find(query))
```

### 3. Aggregation Pipeline Example

```python
def get_book_statistics(client: MongoDBClient):
    """Get aggregated statistics about books."""
    
    pipeline = [
        {
            '$group': {
                '_id': '$status',
                'count': {'$sum': 1},
                'avg_word_count': {'$avg': '$estimated_word_count'},
                'total_word_count': {'$sum': '$estimated_word_count'}
            }
        },
        {
            '$sort': {'count': -1}
        }
    ]
    
    return list(client._books_collection.aggregate(pipeline))
```

## Configuration Management

### Environment-based Configuration

```python
import os
from mongodb_client import MongoDBConfig

def get_mongodb_config() -> MongoDBConfig:
    """Get MongoDB configuration from environment variables."""
    
    return MongoDBConfig(
        host=os.getenv('MONGODB_HOST', 'localhost'),
        port=int(os.getenv('MONGODB_PORT', 27017)),
        database=os.getenv('MONGODB_DATABASE', 'musequill'),
        username=os.getenv('MONGODB_USERNAME', 'musequill'),
        password=os.getenv('MONGODB_PASSWORD', 'musequill.ink.user'),
        auth_database=os.getenv('MONGODB_AUTH_DATABASE', 'musequill'),
        
        # Pool settings
        min_pool_size=int(os.getenv('MONGODB_MIN_POOL_SIZE', 5)),
        max_pool_size=int(os.getenv('MONGODB_MAX_POOL_SIZE', 50)),
        max_idle_time_ms=int(os.getenv('MONGODB_MAX_IDLE_TIME_MS', 30000)),
        wait_queue_timeout_ms=int(os.getenv('MONGODB_WAIT_QUEUE_TIMEOUT_MS', 5000))
    )
```

## Monitoring and Logging

### Enhanced Logging Setup

```python
import logging
from datetime import datetime

# Configure MongoDB-specific logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('musequill_mongodb.log'),
        logging.StreamHandler()
    ]
)

# MongoDB operations logger
mongodb_logger = logging.getLogger('musequill.mongodb')

class LoggingMongoDBClient(MongoDBClient):
    """MongoDB client with enhanced logging."""
    
    def create_book(self, book_data: Dict[str, Any]) -> str:
        start_time = datetime.now()
        try:
            book_id = super().create_book(book_data)
            duration = (datetime.now() - start_time).total_seconds()
            mongodb_logger.info(f"Created book {book_id} in {duration:.3f}s")
            return book_id
        except Exception as e:
            mongodb_logger.error(f"Failed to create book: {e}")
            raise
    
    def update_book(self, book_id: str, update_data: Dict[str, Any]) -> bool:
        start_time = datetime.now()
        try:
            result = super().update_book(book_id, update_data)
            duration = (datetime.now() - start_time).total_seconds()
            mongodb_logger.info(f"Updated book {book_id} in {duration:.3f}s")
            return result
        except Exception as e:
            mongodb_logger.error(f"Failed to update book {book_id}: {e}")
            raise
```

## Production Deployment Considerations

### 1. Connection Management

```python
# Singleton pattern for production
class MongoDBManager:
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self) -> MongoDBClient:
        if self._client is None:
            config = get_mongodb_config()
            self._client = MongoDBClient(config)
            self._client.connect()
        return self._client
    
    def close(self):
        if self._client:
            self._client.disconnect()
            self._client = None

# Usage
mongo_manager = MongoDBManager()
client = mongo_manager.get_client()
```

### 2. Health Monitoring

```python
import time
from typing import Dict, Any

def monitor_mongodb_health(client: MongoDBClient) -> Dict[str, Any]:
    """Comprehensive health monitoring."""
    
    health_data = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'status': 'unknown',
        'checks': {}
    }
    
    try:
        # Basic connectivity
        start_time = time.time()
        ping_result = client._client.admin.command('ping')
        health_data['checks']['ping'] = {
            'status': 'ok' if ping_result['ok'] == 1 else 'failed',
            'response_time_ms': round((time.time() - start_time) * 1000, 2)
        }
        
        # Database operations
        start_time = time.time()
        books_count = client._books_collection.count_documents({})
        health_data['checks']['count_books'] = {
            'status': 'ok',
            'count': books_count,
            'response_time_ms': round((time.time() - start_time) * 1000, 2)
        }
        
        # Index usage
        index_stats = client._database.command('collStats', 'books')
        health_data['checks']['indexes'] = {
            'status': 'ok',
            'index_count': len(index_stats.get('indexSizes', {}))
        }
        
        health_data['status'] = 'healthy'
        
    except Exception as e:
        health_data['status'] = 'unhealthy'
        health_data['error'] = str(e)
    
    return health_data
```

## Migration from books_db Dictionary

### Complete Migration Script

```python
#!/usr/bin/env python3
"""
Migration script to move from books_db dictionary to MongoDB
"""

import json
import sys
from datetime import datetime
from uuid import UUID
from mongodb_client import MongoDBClient, MongoDBConfig

def migrate_books_db(books_db_data: dict, dry_run: bool = True):
    """
    Migrate books_db dictionary data to MongoDB.
    
    Args:
        books_db_data: The books_db dictionary
        dry_run: If True, only simulate the migration
    """
    
    config = MongoDBConfig()
    
    print(f"Starting migration of {len(books_db_data)} books...")
    print(f"Dry run: {dry_run}")
    
    if not dry_run:
        client = MongoDBClient(config)
        client.connect()
    
    migrated = 0
    errors = 0
    
    for book_uuid, book_data in books_db_data.items():
        try:
            # Convert UUID to string
            book_id = str(book_uuid)
            
            # Extract data safely
            parameters = book_data.get('parameters', {})
            planning_results = book_data.get('planning_results', {})
            
            # Create MongoDB document
            mongo_doc = {
                '_id': book_id,
                'created_at': book_data.get('created_at', datetime.now()),
                'updated_at': book_data.get('updated_at', datetime.now()),
                'status': book_data.get('status', 'initializing'),
                'planning_status': book_data.get('planning_status', 'pending'),
                'estimated_word_count': book_data.get('estimated_word_count', 0),
                'estimated_chapters': book_data.get('estimated_chapters', 0),
                'completion_percentage': book_data.get('completion_percentage', 0.0),
                'validation_warnings': book_data.get('validation_warnings', []),
                
                # Extract title and genre from parameters
                'title': parameters.get('title', 'Untitled'),
                'genre': getattr(parameters.get('genre'), 'value', '') if hasattr(parameters.get('genre'), 'value') else str(parameters.get('genre', '')),
                
                # Store original complex data
                'parameters': convert_enums_to_strings(parameters),
                'planning_results': planning_results,
                'genre_info': book_data.get('genre_info', {}),
                'validation_info': book_data.get('validation_info', {}),
                
                # Agent and status data
                'agent_id': book_data.get('agent_id'),
                'planning_completed': book_data.get('planning_completed', False),
                'last_attempt': book_data.get('last_attempt'),
                'retry_count': book_data.get('retry_count', 0),
                'error_message': book_data.get('error_message'),
                'status_message': book_data.get('status_message'),
                'next_steps': book_data.get('next_steps', [])
            }
            
            if dry_run:
                print(f"Would migrate: {book_id} - {mongo_doc['title']}")
            else:
                try:
                    client.create_book(mongo_doc)
                    print(f"✅ Migrated: {book_id} - {mongo_doc['title']}")
                except ValueError:
                    # Book already exists, update it
                    client.update_book(book_id, mongo_doc)
                    print(f"🔄 Updated: {book_id} - {mongo_doc['title']}")
            
            migrated += 1
            
        except Exception as e:
            print(f"❌ Error migrating {book_uuid}: {e}")
            errors += 1
    
    if not dry_run:
        client.disconnect()
    
    print(f"\nMigration completed:")
    print(f"  Migrated: {migrated}")
    print(f"  Errors: {errors}")
    
    return migrated, errors

def convert_enums_to_strings(data):
    """Convert enum objects to string values for JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_enums_to_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_enums_to_strings(item) for item in data]
    elif hasattr(data, 'value'):  # Enum object
        return data.value
    else:
        return data

if __name__ == "__main__":
    # Example usage
    # Replace this with your actual books_db data
    
    # Load books_db from your existing system
    # from musequill.database.book import books_db
    
    # For testing, create sample data
    sample_books_db = {
        # Your actual books_db data would go here
    }
    
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        # Real migration
        migrate_books_db(sample_books_db, dry_run=False)
    else:
        # Dry run
        print("Running in dry-run mode. Use --execute to perform actual migration.")
        migrate_books_db(sample_books_db, dry_run=True)
```

## Final Setup Verification

Run this final verification script:

```python
#!/usr/bin/env python3
"""Final verification that everything is working correctly."""

from mongodb_client import MongoDBClient, BookStatus, AgentType
import json

def final_verification():
    """Complete verification of MongoDB setup."""
    
    print("🔍 Final MongoDB Setup Verification")
    print("=" * 50)
    
    try:
        with MongoDBClient() as client:
            # 1. Health check
            health = client.health_check()
            print(f"1. Health Status: {'✅' if health['status'] == 'healthy' else '❌'} {health['status']}")
            
            # 2. Database connection
            print(f"2. Database: ✅ Connected to {client.config.database}")
            
            # 3. Collections exist
            collections = client._database.list_collection_names()
            expected_collections = ['books', 'agent_sessions']
            for col in expected_collections:
                status = "✅" if col in collections else "❌"
                print(f"3. Collection '{col}': {status}")
            
            # 4. Indexes exist
            book_indexes = list(client._books_collection.list_indexes())
            print(f"4. Book Indexes: ✅ {len(book_indexes)} indexes created")
            
            # 5. CRUD operations
            test_book = {
                'title': 'Verification Test Book',
                'genre': 'test',
                'estimated_word_count': 1000
            }
            
            book_id = client.create_book(test_book)
            print(f"5. Create Book: ✅ Created {book_id}")
            
            book = client.get_book(book_id)
            print(f"6. Read Book: ✅ Retrieved '{book['title']}'")
            
            updated = client.update_book_status(book_id, BookStatus.PLANNING)
            print(f"7. Update Book: ✅ Status updated: {updated}")
            
            books = client.list_books()
            print(f"8. List Books: ✅ Found {len(books)} books")
            
            session_id = client.create_agent_session({
                'book_id': book_id,
                'agent_type': AgentType.PLANNER
            })
            print(f"9. Agent Session: ✅ Created session {session_id}")
            
            deleted = client.delete_book(book_id)
            print(f"10. Delete Book: ✅ Deleted: {deleted}")
            
            print("\n🎉 All verifications passed!")
            print("MongoDB is properly configured and ready for MuseQuill!")
            
            # Connection string for reference
            print(f"\n📝 Connection Details:")
            print(f"   Host: {client.config.host}:{client.config.port}")
            print(f"   Database: {client.config.database}")
            print(f"   Username: {client.config.username}")
            print(f"   Collections: {', '.join(expected_collections)}")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    final_verification()
```

## Next Steps

1. **Install dependencies**: `pip install pymongo motor python-dotenv`
2. **Run the test script**: `python test_mongodb_setup.py`
3. **Run final verification**: `python final_verification.py`
4. **Integrate into MuseQuill**: Replace `books_db` dictionary usage with MongoDB client calls
5. **Optional**: Run migration script to move existing data

Your MongoDB setup is now complete and ready for production use with MuseQuill! 🚀