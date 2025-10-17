"""
Shared aiohttp Session Manager

Production-grade pattern: ONE shared session for the entire application.
This session is created at startup and reused for ALL HTTP requests.

Benefits:
- Connection reuse via HTTP keep-alive
- Single connection pool managed by aiohttp
- DNS caching across all requests
- No file descriptor leaks
- Minimal overhead
"""

import asyncio
import logging
from typing import Optional
import aiohttp

logger = logging.getLogger(__name__)


class SharedSessionManager:
    """
    Manages a single shared aiohttp ClientSession for the entire application.
    
    This pattern uses ONE session shared across all requests,
    enabling massive connection reuse and preventing file descriptor leaks.
    """
    
    def __init__(
        self,
        connector_limit: int = 0,  # 0 = unlimited
        keepalive_timeout: int = 120,  # 2 minutes
        ttl_dns_cache: int = 300,  # 5 minutes
    ):
        self.connector_limit = connector_limit
        self.keepalive_timeout = keepalive_timeout
        self.ttl_dns_cache = ttl_dns_cache
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """
        Initialize the shared session.
        Called once at application startup.
        
        Uses optimal production settings for connection pooling.
        """
        async with self._lock:
            if self._initialized:
                logger.warning("Shared session already initialized")
                return
            
            # Create connector with production-grade settings
            self._connector = aiohttp.TCPConnector(
                limit=self.connector_limit,
                keepalive_timeout=self.keepalive_timeout,
                ttl_dns_cache=self.ttl_dns_cache,
                enable_cleanup_closed=True,
            )
            
            # Create session (no session-level timeout to avoid conflicts with per-request timeouts)
            self._session = aiohttp.ClientSession(connector=self._connector)
            
            self._initialized = True
            
            logger.info(
                f"SESSION REUSE: Created shared aiohttp session for connection pooling "
                f"(limit={self.connector_limit}, keepalive={self.keepalive_timeout}s, "
                f"dns_cache={self.ttl_dns_cache}s)"
            )
    
    async def shutdown(self):
        """
        Shutdown the shared session.
        Called once at application shutdown.
        """
        async with self._lock:
            if not self._initialized:
                return
            
            logger.info("Closing shared aiohttp session...")
            
            if self._session:
                await self._session.close()
                # Give aiohttp time to close connections gracefully
                await asyncio.sleep(0.25)
                self._session = None
            
            if self._connector:
                await self._connector.close()
                self._connector = None
            
            self._initialized = False
            logger.info("Shared aiohttp session closed")
    
    def get_session(self) -> aiohttp.ClientSession:
        """
        Get the shared session for making HTTP requests.
        
        Returns:
            The shared aiohttp ClientSession
            
        Raises:
            RuntimeError: If session not initialized
        """
        if not self._initialized or self._session is None:
            raise RuntimeError(
                "Shared session not initialized. "
                "Call initialize() first (usually in app startup)."
            )
        
        if self._session.closed:
            raise RuntimeError("Shared session is closed")
        
        return self._session
    
    @property
    def is_initialized(self) -> bool:
        """Check if session is initialized."""
        return self._initialized and self._session is not None


# Global instance - created once at module import
_global_session_manager = SharedSessionManager()


def get_session_manager() -> SharedSessionManager:
    """Get the global session manager instance."""
    return _global_session_manager


async def get_shared_session() -> Optional[aiohttp.ClientSession]:
    """
    Get the shared aiohttp session (convenience function).
    
    Returns:
        The shared session if initialized, None otherwise
    """
    try:
        return _global_session_manager.get_session()
    except RuntimeError:
        logger.warning("Shared session not available")
        return None

