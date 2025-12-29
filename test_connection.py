import os
from dotenv import load_dotenv
from github import Github
from github.Auth import Token

# Load env
load_dotenv()
token = os.getenv("GITHUB_TOKEN")

if not token:
    print("❌ ERROR: GITHUB_TOKEN not found. Check your .env file")
else:
    try:
        # ✅ Modern authentication
        auth = Token(token)
        g = Github(auth=auth)

        user = g.get_user().login
        print(f"✅ SUCCESS: Connected as {user}")

        # ✅ Correct rate limit access
        rate = g.get_rate_limit()
        print(
            f"📊 API Status: "
            f"Core={rate.resources.core.remaining}, "
            f"Search={rate.resources.search.remaining}"
        )

    except Exception as e:
        print(f"❌ ERROR: {e}")
