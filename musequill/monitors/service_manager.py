"""
Monitor Service Manager - Centralized management for all monitoring components

This module provides a unified interface for starting, stopping, and managing
all monitoring services including BookRetriever and BookMonitor.
"""

import asyncio
import logging
import signal
import threading
import time
import atexit
from typing import Dict, List, Optional, Any
from datetime import datetime

from musequill.config.logging import get_logger
from musequill.monitors.book_retriever import BookRetriever
from musequill.monitors.book_retriever_config import BookRetrieverConfig
from musequill.monitors.book_monitor import BookMonitor

logger = get_logger(__name__)


class MonitorServiceManager:
    """
    Centralized manager for all monitoring services in the MuseQuill application.
    
    This class handles:
    - Starting and stopping all monitor services
    - Health checking across services
    - Graceful shutdown coordination
    - Service status reporting
    """
    
    def __init__(self):
        self.running = False
        self.services: Dict[str, Any] = {}
        self.shutdown_event = threading.Event()
        
        # Initialize services
        self._initialize_services()
        
        # Register shutdown handlers
        atexit.register(self.stop_all)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info("Monitor Service Manager initialized")
    
    def _initialize_services(self):
        """Initialize all monitoring services."""
        try:
            # Initialize BookRetriever
            book_retriever_config = BookRetrieverConfig()
            self.services['book_retriever'] = {
                'instance': BookRetriever(config=book_retriever_config),
                'name': 'Book Retriever',
                'description': 'Retrieves books from Redis queue and manages orchestrations',
                'status': 'initialized',
                'started_at': None,
                'error': None
            }
            
            # Initialize BookMonitor
            self.services['book_monitor'] = {
                'instance': BookMonitor(),
                'name': 'Book Pipeline Monitor',
                'description': 'Monitors book processing pipeline and manages state transitions',
                'status': 'initialized',
                'started_at': None,
                'error': None
            }
            
            logger.info(f"Initialized {len(self.services)} monitoring services")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring services: {e}")
            raise
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating monitor services shutdown...")
        self.stop_all()
    
    def start_all(self) -> bool:
        """
        Start all monitoring services.
        
        Returns:
            bool: True if all services started successfully, False otherwise
        """
        if self.running:
            logger.warning("Monitor services are already running")
            return True
        
        logger.info("Starting all monitoring services...")
        
        success_count = 0
        total_services = len(self.services)
        
        for service_name, service_info in self.services.items():
            try:
                logger.info(f"Starting {service_info['name']}...")
                
                # Start the service
                service_info['instance'].start()
                
                # Update service status
                service_info['status'] = 'running'
                service_info['started_at'] = datetime.now()
                service_info['error'] = None
                
                success_count += 1
                logger.info(f"✅ {service_info['name']} started successfully")
                
                # Give services time to initialize
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Failed to start {service_info['name']}: {e}")
                service_info['status'] = 'failed'
                service_info['error'] = str(e)
        
        self.running = success_count > 0
        
        if success_count == total_services:
            logger.info(f"🚀 All {total_services} monitoring services started successfully")
            return True
        elif success_count > 0:
            logger.warning(f"⚠️ Partial success: {success_count}/{total_services} services started")
            return True
        else:
            logger.error("❌ Failed to start any monitoring services")
            return False
    
    def stop_all(self, timeout: float = 30.0) -> None:
        """
        Stop all monitoring services gracefully.
        
        Args:
            timeout: Maximum time to wait for services to stop
        """
        if not self.running:
            logger.info("Monitor services are not running")
            return
        
        logger.info("Stopping all monitoring services...")
        
        self.shutdown_event.set()
        self.running = False
        
        for service_name, service_info in self.services.items():
            if service_info['status'] == 'running':
                try:
                    logger.info(f"Stopping {service_info['name']}...")
                    
                    # Stop the service with timeout
                    service_info['instance'].stop(timeout=timeout)
                    
                    service_info['status'] = 'stopped'
                    service_info['error'] = None
                    
                    logger.info(f"✅ {service_info['name']} stopped successfully")
                    
                except Exception as e:
                    logger.error(f"❌ Error stopping {service_info['name']}: {e}")
                    service_info['status'] = 'error'
                    service_info['error'] = str(e)
        
        logger.info("Monitor services shutdown completed")
    
    def get_service_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of monitoring services.
        
        Args:
            service_name: Optional specific service name to check
            
        Returns:
            Dict containing service status information
        """
        if service_name:
            if service_name not in self.services:
                return {"error": f"Service '{service_name}' not found"}
            
            service_info = self.services[service_name]
            status_data = {
                "name": service_info['name'],
                "description": service_info['description'],
                "status": service_info['status'],
                "started_at": service_info['started_at'].isoformat() if service_info['started_at'] else None,
                "error": service_info['error']
            }
            
            # Get detailed status from the service instance if running
            if service_info['status'] == 'running':
                try:
                    detailed_status = service_info['instance'].get_status()
                    status_data['details'] = detailed_status
                except Exception as e:
                    status_data['details_error'] = str(e)
            
            return status_data
        
        # Return status for all services
        all_status = {
            "manager_running": self.running,
            "total_services": len(self.services),
            "services": {}
        }
        
        running_count = 0
        for service_name, service_info in self.services.items():
            status_data = {
                "name": service_info['name'],
                "description": service_info['description'],
                "status": service_info['status'],
                "started_at": service_info['started_at'].isoformat() if service_info['started_at'] else None,
                "error": service_info['error']
            }
            
            if service_info['status'] == 'running':
                running_count += 1
                try:
                    detailed_status = service_info['instance'].get_status()
                    status_data['details'] = detailed_status
                except Exception as e:
                    status_data['details_error'] = str(e)
            
            all_status['services'][service_name] = status_data
        
        all_status['running_services'] = running_count
        all_status['health'] = 'healthy' if running_count == len(self.services) else 'degraded' if running_count > 0 else 'unhealthy'
        
        return all_status
    
    def restart_service(self, service_name: str, timeout: float = 10.0) -> bool:
        """
        Restart a specific monitoring service.
        
        Args:
            service_name: Name of the service to restart
            timeout: Timeout for stop operation
            
        Returns:
            bool: True if restart was successful
        """
        if service_name not in self.services:
            logger.error(f"Service '{service_name}' not found")
            return False
        
        service_info = self.services[service_name]
        
        try:
            logger.info(f"Restarting {service_info['name']}...")
            
            # Stop the service if running
            if service_info['status'] == 'running':
                service_info['instance'].stop(timeout=timeout)
            
            # Start the service
            service_info['instance'].start()
            
            # Update status
            service_info['status'] = 'running'
            service_info['started_at'] = datetime.now()
            service_info['error'] = None
            
            logger.info(f"✅ {service_info['name']} restarted successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to restart {service_info['name']}: {e}")
            service_info['status'] = 'failed'
            service_info['error'] = str(e)
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check of all monitoring services.
        
        Returns:
            Dict containing health check results
        """
        health_data = {
            "overall_status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "manager_running": self.running,
            "services": {}
        }
        
        healthy_services = 0
        total_services = len(self.services)
        
        for service_name, service_info in self.services.items():
            service_health = {
                "status": service_info['status'],
                "healthy": False,
                "error": service_info['error']
            }
            
            if service_info['status'] == 'running':
                try:
                    # Try to get status from the service (basic health check)
                    service_status = service_info['instance'].get_status()
                    service_health['healthy'] = service_status.get('running', False)
                    service_health['details'] = service_status
                    
                    if service_health['healthy']:
                        healthy_services += 1
                        
                except Exception as e:
                    service_health['error'] = f"Health check failed: {e}"
                    service_health['healthy'] = False
            
            health_data['services'][service_name] = service_health
        
        # Determine overall health
        if healthy_services == total_services:
            health_data['overall_status'] = 'healthy'
        elif healthy_services > 0:
            health_data['overall_status'] = 'degraded'
        else:
            health_data['overall_status'] = 'unhealthy'
        
        health_data['healthy_services'] = healthy_services
        health_data['total_services'] = total_services
        
        return health_data


# Global instance
_monitor_service_manager: Optional[MonitorServiceManager] = None


def get_monitor_service_manager() -> MonitorServiceManager:
    """
    Get the global monitor service manager instance.
    
    Returns:
        MonitorServiceManager: The global instance
    """
    global _monitor_service_manager
    
    if _monitor_service_manager is None:
        _monitor_service_manager = MonitorServiceManager()
    
    return _monitor_service_manager


def start_monitoring_services() -> bool:
    """
    Start all monitoring services using the global manager.
    
    Returns:
        bool: True if services started successfully
    """
    manager = get_monitor_service_manager()
    return manager.start_all()


def stop_monitoring_services(timeout: float = 30.0) -> None:
    """
    Stop all monitoring services using the global manager.
    
    Args:
        timeout: Maximum time to wait for services to stop
    """
    global _monitor_service_manager
    
    if _monitor_service_manager:
        _monitor_service_manager.stop_all(timeout=timeout)


def get_monitoring_status() -> Dict[str, Any]:
    """
    Get status of all monitoring services.
    
    Returns:
        Dict containing service status information
    """
    manager = get_monitor_service_manager()
    return manager.get_service_status()