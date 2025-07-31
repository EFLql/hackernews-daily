import time
import requests
from functools import wraps

def retry_request(max_retries=3, delay=1):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
        return wrapper
    return decorator

@retry_request(max_retries=3)
def fetch_url(url, headers=None, timeout=10):
    return requests.get(url, headers=headers, timeout=timeout)