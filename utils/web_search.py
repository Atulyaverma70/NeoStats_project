import requests
from config import config

SERPAPI_KEY = getattr(config, "SERPAPI_KEY", None)
SERPAPI_ENDPOINT = "https://serpapi.com/search.json"  # SerpAPI JSON endpoint

def web_search(query):
    """Perform web search via SerpAPI"""
    if not SERPAPI_KEY:
        return "SERPAPI_KEY not set"
    
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",  # Using Google engine in SerpAPI
        "num": 3             # Number of results
    }

    try:
        response = requests.get(SERPAPI_ENDPOINT, params=params)
        response.raise_for_status()
        results = response.json()

        snippets = []
        for item in results.get("organic_results", []):
            snippets.append(item.get("snippet", ""))

        if not snippets:
            return "No results found"
        return "\n".join(snippets)

    except Exception as e:
        return f"Web search error: {str(e)}"
