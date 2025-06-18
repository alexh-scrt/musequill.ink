from uuid import UUID
from typing import Dict, List, Any
from musequill.database.book_db import MongoDBClient, MongoDBConfig
from musequill.config.logging import get_logger


logger = get_logger(__name__)
# Create global client instance
mongodb_config = MongoDBConfig()
logger.debug(f'MongoDBConfig: {mongodb_config.to_dict()}')


# Update functions to use MongoDB instead of dictionary
def create_book(book_data):
    """ Create a new book record given the Book Create Request """
    try:
        with MongoDBClient(mongodb_config) as client:
            book_id = client.create_book(book_data)
            if not book_id:
                raise ValueError(f"Book {book_id} not found")
            
    except ValueError as e:
        logger.error(f"Book operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Handle database connectivity issues
        raise


def get_book(book_id) -> Dict:
    """ Get book given its uuid """
    try:
        with MongoDBClient(mongodb_config) as client:
            book = client.get_book(book_id)
            if not book:
                raise ValueError(f"Book with id:{book_id} not found")
            return book
    except ValueError as e:
        logger.error(f"Book operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Handle database connectivity issues
        raise

def list_books() -> List[Dict[str, Any]]:
    """ Get book given its uuid """
    try:
        with MongoDBClient(mongodb_config) as client:
            books = client.list_books()
            if not books:
                raise ValueError(f"Books not found")
            return books
    except ValueError as e:
        logger.error(f"Book operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Handle database connectivity issues
        raise

def book_exists(book_id) -> bool:
    """ Get book given its uuid """
    try:
        with MongoDBClient(mongodb_config) as client:
            exists = client.book_exists(book_id)
            return exists
    except ValueError as e:
        logger.error(f"Book operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Handle database connectivity issues
        raise

def update_book(book_id, data):
    """" Update book data given the book id """
    try:
        with MongoDBClient(mongodb_config) as client:
            rc = client.update_book(book_id, data)
            if not rc:
                raise ValueError(f"Book with id:{book_id} not updated")
            
    except ValueError as e:
        logger.error(f"Book operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Handle database connectivity issues
        raise


# In musequill/agents/integration.py
def update_book_status(book_id: UUID, status: str, additional_data: dict = None):
    """Update book status in MongoDB instead of books_db."""
    try:
        with MongoDBClient(mongodb_config) as client:
            rc = client.update_book_status(str(book_id), status, additional_data)
            if not rc:
                raise ValueError(f"Book with id:{book_id} status update failed")
            
    except ValueError as e:
        logger.error(f"Book operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        # Handle database connectivity issues
        raise