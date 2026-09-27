import requests


def verify_reference(url: str):
    try:
        response = requests.get(url, timeout=10, allow_redirects=True)
        if 200 <= response.status_code < 400:
            return f"SUCCESS: {url} is a valid reference."
        return f"FAILED: {url} is not a valid reference (HTTP {response.status_code})."
    except requests.RequestException as e:
        return f"FAILED: {url} could not be verified ({type(e).__name__}: {e})."