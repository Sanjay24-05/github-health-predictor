import time
import logging
import os
from github import Github, GithubException
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubClient:
    def __init__(self):
        load_dotenv()
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN not found in .env file.")
        self.client = Github(token)

    def handle_rate_limit(self):
        """Checks remaining rate limit and sleeps if it is too low."""
        rate_limit = self.client.get_rate_limit().core
        if rate_limit.remaining < 10:  # Threshold for safety
            reset_timestamp = rate_limit.reset.timestamp()
            sleep_time = max(reset_timestamp - time.time(), 0) + 5
            logger.warning(f"Rate limit low. Sleeping {sleep_time:.0f}s...")
            time.sleep(sleep_time)

    def search_repositories(self, query, limit=100):
        """Searches for repositories based on a query string."""
        self.handle_rate_limit()
        return self.client.search_repositories(query=query)[:limit]