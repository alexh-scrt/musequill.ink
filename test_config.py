#!/usr/bin/env python3
"""Test script to check if BookMonitorConfig reads environment variables."""

import os
from musequill.monitors.book_monitor_config import BookMonitorConfig

def test_env_loading():
    print("=== BookMonitorConfig Environment Variable Test ===")
    
    # First, check what environment variables are actually set from .env
    env_vars = [
        'MONGODB_URL', 'MONGODB_DATABASE', 'MONGODB_COLLECTION',
        'MONGODB_USERNAME', 'MONGODB_PASSWORD',
        'REDIS_URL', 'BOOK_QUEUE_NAME', 'BOOK_POLL_INTERVAL', 'MAX_BOOKS_PER_BATCH'
    ]
    
    print("\nCurrent environment variables:")
    for var in env_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"  {var}: {value}")
    
    print("\n=== Creating BookMonitorConfig instance ===")
    config = BookMonitorConfig()
    
    print(f"mongodb_url: {config.mongodb_url}")
    print(f"database_name: {config.database_name}")
    print(f"collection_name: {config.collection_name}")
    print(f"database_username: {config.database_username}")
    print(f"database_password: {config.database_password}")
    print(f"redis_url: {config.redis_url}")
    print(f"queue_name: {config.queue_name}")
    print(f"poll_interval: {config.poll_interval}")
    print(f"max_books_per_batch: {config.max_books_per_batch}")
    
    print("\n=== Analysis ===")
    issues = []
    
    # Check if values match environment
    if os.environ.get('MONGODB_URL') and config.mongodb_url != os.environ.get('MONGODB_URL'):
        issues.append(f"MONGODB_URL mismatch: env={os.environ.get('MONGODB_URL')}, config={config.mongodb_url}")
    
    if os.environ.get('REDIS_URL') and config.redis_url != os.environ.get('REDIS_URL'):
        issues.append(f"REDIS_URL mismatch: env={os.environ.get('REDIS_URL')}, config={config.redis_url}")
    
    if os.environ.get('BOOK_POLL_INTERVAL') and config.poll_interval != int(os.environ.get('BOOK_POLL_INTERVAL')):
        issues.append(f"BOOK_POLL_INTERVAL mismatch: env={os.environ.get('BOOK_POLL_INTERVAL')}, config={config.poll_interval}")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All environment variables are being read correctly")

if __name__ == "__main__":
    # First import bootstrap to ensure environment is loaded
    import bootstrap
    test_env_loading()