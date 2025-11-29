import functools
import logging

# Configure basic logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def safe_execution(log_msg: str, return_on_error=None):
    """
    Decorator to wrap functions in a try-except block.
    
    Args:
        log_msg (str): The custom error message prefix.
        return_on_error (Any): Value to return if an exception occurs.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"{log_msg}: {e}")
                # Optional: Log to file using logging module
                # logging.error(f"{log_msg}: {e}") 
                return return_on_error
        return wrapper
    return decorator