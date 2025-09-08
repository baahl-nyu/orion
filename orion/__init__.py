# orion/__init__.py

from .core import (
    init_scheme, delete_scheme, encode, decode, encrypt, decrypt, fit, compile
)

from .core.logger import logger, set_log_level, TracerDashboard

__version__ = "1.0.1"

__all__ = [
    "init_scheme", "delete_scheme", "encode", "decode",
    "encrypt", "decrypt", "fit", "compile",
    "logger", "set_log_level", "Dashboard",
]