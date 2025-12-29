# src/data/github_client.py

import os
import time
import logging
from github import Github, GithubException
from github.Auth import Token
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubClient:
    def __init__(self):
        load_dotenv()
        token = os.getenv("GITHUB_TOKEN")

        if not token:
            raise ValueError("GITHUB_TOKEN not found in environment.")

        self.client = Github(auth=Token(token))

    def _handle_rate_limit(self):
        rate = self.client.get_rate_limit().resources.core

        if rate.remaining < 10:
            sleep_time = max(rate.reset.timestamp() - time.time(), 0) + 5
            logger.warning(f"Rate limit low. Sleeping {sleep_time:.0f}s...")
            time.sleep(sleep_time)

    def search_repositories(self, query, limit=100):
        results = self.client.search_repositories(query=query)
        repos = []

        try:
            for repo in results:
                self._handle_rate_limit()
                repos.append(repo)

                if len(repos) >= limit:
                    break

        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            raise

        return repos
