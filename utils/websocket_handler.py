"""
WebSocket Connection Stability Handler
Prevents crashes from disconnected clients during long operations
"""

import functools
import asyncio
import tornado.websocket
from typing import Callable, Any
import logging

# Suppress WebSocket errors from flooding logs
logging.getLogger('tornado.websocket').setLevel(logging.CRITICAL)
logging.getLogger('tornado.application').setLevel(logging.CRITICAL)


def safe_websocket_operation(func: Callable) -> Callable:
    """
    Decorator to safely handle WebSocket operations
    Ignores WebSocketClosedError silently
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (tornado.websocket.WebSocketClosedError, 
                tornado.iostream.StreamClosedError,
                BrokenPipeError,
                ConnectionResetError):
            # Client disconnected - ignore silently
            return None
        except Exception as e:
            # Log other errors but don't crash
            logging.debug(f"WebSocket error in {func.__name__}: {e}")
            return None
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (tornado.websocket.WebSocketClosedError,
                tornado.iostream.StreamClosedError,
                BrokenPipeError,
                ConnectionResetError):
            return None
        except Exception as e:
            logging.debug(f"WebSocket error in {func.__name__}: {e}")
            return None
    
    # Return appropriate wrapper based on function type
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class SafeWebSocketMixin:
    """
    Mixin for WebSocket handlers to safely handle disconnections
    """
    
    def safe_write(self, message: str) -> bool:
        """Safely write to WebSocket, return False if closed"""
        try:
            if self.ws_connection and not self.ws_connection.is_closing():
                self.write_message(message)
                return True
        except (tornado.websocket.WebSocketClosedError,
                tornado.iostream.StreamClosedError):
            pass
        return False
    
    def on_close(self):
        """Override to handle close gracefully"""
        # Set flag to prevent further operations
        self._closed = True
        # Call parent if exists
        if hasattr(super(), 'on_close'):
            super().on_close()


def patch_streamlit_websocket():
    """
    Monkey-patch Streamlit's WebSocket handler to be more resilient
    Call this at app startup
    """
    try:
        from streamlit.web.server.browser_websocket_handler import BrowserWebSocketHandler
        
        # Store original methods
        original_write = BrowserWebSocketHandler.write_message
        original_on_close = BrowserWebSocketHandler.on_close
        
        def safe_write_message(self, message, binary=False):
            """Safe write that ignores closed connections"""
            try:
                # Check if connection is still open
                if hasattr(self, 'ws_connection') and self.ws_connection:
                    if not self.ws_connection.is_closing():
                        return original_write(self, message, binary=binary)
            except (tornado.websocket.WebSocketClosedError,
                    tornado.iostream.StreamClosedError,
                    AttributeError):
                pass
            return None
        
        def safe_on_close(self, code=None, reason=None):
            """Safe close handler"""
            try:
                # Mark as closed to prevent further operations
                self._is_closed = True
                return original_on_close(self, code, reason)
            except Exception:
                pass
        
        # Apply patches
        BrowserWebSocketHandler.write_message = safe_write_message
        BrowserWebSocketHandler.on_close = safe_on_close
        BrowserWebSocketHandler._is_closed = False
        
    except ImportError:
        # Streamlit version may differ, ignore
        pass


def configure_tornado_logging():
    """Configure Tornado to be less verbose about WebSocket errors"""
    import logging
    
    # Get loggers
    access_log = logging.getLogger("tornado.access")
    app_log = logging.getLogger("tornado.application")
    gen_log = logging.getLogger("tornado.general")
    
    # Set levels
    access_log.setLevel(logging.WARNING)
    app_log.setLevel(logging.ERROR)
    gen_log.setLevel(logging.WARNING)
    
    # Filter out specific WebSocket error patterns
    class WebSocketErrorFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            blocked_patterns = [
                'WebSocketClosedError',
                'StreamClosedError',
                'ConnectionResetError',
                'BrokenPipeError',
                'Task exception was never retrieved'
            ]
            return not any(p in message for p in blocked_patterns)
    
    # Apply filter
    for handler in logging.root.handlers:
        handler.addFilter(WebSocketErrorFilter())