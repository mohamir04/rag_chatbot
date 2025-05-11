import asyncio
from datetime import datetime, timedelta
import re
import httpx
from bs4 import BeautifulSoup

class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                print(f"Rate limiter active: waiting for {wait_time:.2f} seconds.")
                await asyncio.sleep(wait_time)

        self.requests.append(now) 

class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)

    async def fetch_and_parse(self, url: str) -> str:
        """Fetch and parse content from a webpage"""
        try:
            await self.rate_limiter.acquire()

            print(f"Fetching content from: {url}")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    },
                    follow_redirects=True,
                    timeout=30.0, # seconds
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            for element in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                element.decompose()

            text = soup.get_text(separator=' ', strip=True)
            
            text = re.sub(r'\s+', ' ', text).strip()

            # MAX_TEXT_LENGTH = 8000 
            # if len(text) > MAX_TEXT_LENGTH:
            #     text = text[:MAX_TEXT_LENGTH] + "... [content truncated]"

            print(f"Successfully fetched and parsed content ({len(text)} characters) from {url}")
            return text

        except httpx.TimeoutException:
            print(f"Request timed out for URL: {url}")
            return "Error: The request timed out while trying to fetch the webpage."
        except httpx.RequestError as e: # More general network/request error
            print(f"Request error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage. Network issue or invalid URL ({str(e)})"
        except httpx.HTTPStatusError as e: # For 4xx/5xx responses
            print(f"HTTP status error occurred while fetching {url}: {str(e)}")
            return f"Error: Could not access the webpage. Status code: {e.response.status_code} ({str(e)})"
        except Exception as e:
            print(f"Error fetching content from {url}: {type(e).__name__} - {str(e)}")
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"

# Example usage (optional, for testing)
async def main_test():
    fetcher = WebContentFetcher()
    content = await fetcher.fetch_and_parse("https://example.com")
    if not content.startswith("Error:"):
        print("\nFetched Content Snippet:")
        print(content[:500])
    else:
        print(f"\n{content}")

if __name__ == '__main__':
    asyncio.run(main_test()) 