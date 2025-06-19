from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__name__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import bootstrap

from musequill.monitors.book_monitor import BookPipelineMonitor

def main():
    """
    Test function for BookPipelineMonitor.
    Creates a monitor instance, starts it, runs for a few seconds, then stops it.
    """
    
    print("Testing BookPipelineMonitor...")
    
    # Create monitor instance
    monitor = BookPipelineMonitor()
    
    try:
        # Start the monitor
        print("Starting monitor...")
        monitor.start()
        
        # Let it run for 10 seconds
        print("Monitor running for 10 seconds...")
        time.sleep(10)
        
        # Stop the monitor
        print("Stopping monitor...")
        monitor.stop(timeout=5.0)
        
        print("Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, stopping monitor...")
        monitor.stop(timeout=5.0)
    except Exception as e:
        print(f"Error during test: {e}")
        monitor.stop(timeout=5.0)


if __name__ == "__main__":
    main()