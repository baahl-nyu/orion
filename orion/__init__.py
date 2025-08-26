import logging

# Import core functions
from .core import (
    init_scheme,
    delete_scheme, 
    encode, 
    decode, 
    encrypt, 
    decrypt, 
    fit, 
    compile
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

def set_log_level(level):
    """
    Sets the log level.
    Args:
        level (str): DEBUG, INFO, WARNING, ERROR, CRITICAL
    """
    logger.setLevel(level)
    # Also update the 'orion' logger used in your modules
    orion_logger = logging.getLogger("orion")
    orion_logger.setLevel(level)
    
    # Create handler for orion logger if it doesn't have one
    if not orion_logger.handlers:
        orion_handler = logging.StreamHandler()
        orion_handler.setFormatter(formatter)
        orion_logger.addHandler(orion_handler)
        orion_logger.propagate = False

__version__ = "1.0.1"

__all__ = [
    'init_scheme',
    'delete_scheme',
    'encode',
    'decode',
    'encrypt',
    'decrypt',
    'fit',
    'compile',
    'set_log_level',
]