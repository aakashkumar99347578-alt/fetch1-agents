from langchain.tools import tool 
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os 
from dotenv import load_dotenv

load_dotenv()

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not configured in .env."
    try:
        tavily = TavilyClient(api_key=api_key)
        results = tavily.search(query=query, max_results=5)
        out = []
        for r in results.get("results", []):
            title = r.get("title", "No Title")
            url = r.get("url", "")
            snippet = r.get("content", "")[:300]
            out.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}\n")
        if not out:
            return f"No search results found for: {query}"
        return "\n----\n".join(out)
    except Exception as e:
        return f"Error executing search for '{query}': {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    if not url or not url.startswith(("http://", "https://")):
        return f"Could not scrape invalid URL: {url}"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, timeout=10, headers=headers)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:3000] if text else "Page contained no readable text."
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

