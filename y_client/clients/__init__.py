try:
    # try the web-based implementation first
    from .client_web import *
except ImportError:
    # fallback to base implementation
    from .client_base import *
