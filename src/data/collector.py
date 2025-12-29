import pandas as pd
from github_client import GitHubClient
from tqdm import tqdm

def collect_data():
    client = GitHubClient()
    # We search for popular Python repos + some abandoned ones for balance
    queries = [
        "language:python stars:>1000",
        "language:python stars:100..500 archived:true"
    ]
    
    repo_list = []
    
    for q in queries:
        print(f"Executing search: {q}")
        repos = client.search_repositories(q, limit=500)
        
        for repo in tqdm(repos, total=repos.totalCount):
            repo_list.append({
                "full_name": repo.full_name,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "open_issues": repo.open_issues_count,
                "last_update": repo.updated_at,
                "is_archived": repo.archived,
                "description": repo.description
            })
            
    df = pd.DataFrame(repo_list)
    df.to_csv("data/raw/repositories_raw.csv", index=False)
    print("✅ Collection complete! File saved to data/raw/repositories_raw.csv")

if __name__ == "__main__":
    collect_data()