"""
Finds real comments in r/LocalLLM and r/Ollama that describe the exact pain
Ramer targets (manual GPU/CPU layer tuning, VRAM-fit struggles, multi-agent
GPU thrashing), and drafts a short, non-pitchy reply for each using a local
model through Ramer's own proxy.

Read-only. Never posts anything. Writes a review file you read and copy
from by hand -- see the "how do I get people to try this" discussion this
came out of: replies work because a human decided they were worth sending,
not because a bot found a keyword.

Usage:
    python tools/reddit_leads.py
    python tools/reddit_leads.py --posts-per-sub 40 --out tools/leads.md
"""
import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request

SUBREDDITS = ["LocalLLM", "Ollama"]
USER_AGENT = "ramer-lead-finder/0.1 (personal research tool, read-only)"
RAMER_URL = "http://127.0.0.1:11435/v1/chat/completions"
DRAFT_MODEL = "qwen2.5:14b-instruct"
CREDENTIALS_FILE = os.path.join(os.path.dirname(__file__), "reddit_credentials.json")

# Reddit now requires auth even for the read-only .json endpoints (anonymous
# requests get a 302 to /login with reason=lor2). This uses Reddit's
# app-only OAuth grant -- no password, no user session, just a free
# "script" app's client_id/secret from reddit.com/prefs/apps -- which is
# the sanctioned way to read public subreddit/comment data without a user
# login in the loop.

# High-signal phrases -- specific enough that a hit usually means the person
# is actually dealing with GPU/CPU placement, not just mentioning a model.
KEYWORDS = [
    "num_gpu", "n-cpu-moe", "-ngl", "gpu layers", "offload to cpu",
    "cpu offload", "doesn't fit in vram", "won't fit in vram",
    "out of vram", "vram is full", "not enough vram", "ran out of vram",
    "switching models", "juggling models", "multiple agents", "keep_alive",
    "context window ate", "tuning ngl", "hand-tuning", "manually tuning",
]

REPLY_SYSTEM_PROMPT = (
    "You write short, honest Reddit replies. You are NOT selling anything. "
    "You reply only to what the person actually said, referencing a specific "
    "detail from their comment. You mention a tool called Ramer (a proxy that "
    "sits in front of Ollama and auto-tunes the GPU/CPU layer split) only as "
    "something you built because you had the same problem -- never as a pitch. "
    "End by asking if it would actually help their specific case, not by "
    "asking them to try it. Two to four sentences. Lowercase, casual, no "
    "corporate tone, no emoji, no bullet points."
)


def load_credentials() -> tuple:
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, encoding="utf-8") as f:
            creds = json.load(f)
        return creds["client_id"], creds["client_secret"]
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret
    raise SystemExit(
        "No Reddit API credentials found. Reddit now requires auth even for "
        "read-only access. Create a free 'script' app at "
        "https://www.reddit.com/prefs/apps (takes ~2 minutes, no special "
        "permissions needed since this only reads public data), then either:\n"
        f"  - save {{\"client_id\": ..., \"client_secret\": ...}} to {CREDENTIALS_FILE}\n"
        "  - or set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET env vars"
    )


def get_oauth_token() -> str:
    client_id, client_secret = load_credentials()
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token",
        data=b"grant_type=client_credentials",
        headers={
            "Authorization": f"Basic {basic}",
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def http_get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_listing(subreddit: str, limit: int, token: str) -> list:
    url = f"https://oauth.reddit.com/r/{subreddit}/hot?limit={limit}"
    data = http_get_json(url, token)
    return [child["data"] for child in data["data"]["children"] if child["kind"] == "t3"]


def fetch_comments(subreddit: str, post_id: str, token: str) -> list:
    url = f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}?limit=200"
    data = http_get_json(url, token)
    comments = []

    def walk(node):
        if node.get("kind") != "t1":
            return
        d = node["data"]
        body = d.get("body", "")
        if body and body != "[deleted]" and body != "[removed]":
            comments.append(d)
        replies = d.get("replies")
        if isinstance(replies, dict):
            for child in replies.get("data", {}).get("children", []):
                walk(child)

    if len(data) > 1:
        for child in data[1]["data"]["children"]:
            walk(child)
    return comments


def score_text(text: str) -> list:
    lowered = text.lower()
    return [kw for kw in KEYWORDS if kw in lowered]


def draft_reply(post_title: str, comment_body: str) -> str:
    user_prompt = (
        f"Reddit post title: {post_title!r}\n"
        f"Comment I'm replying to: {comment_body!r}\n\n"
        "Draft my reply."
    )
    payload = {
        "model": DRAFT_MODEL,
        "messages": [
            {"role": "system", "content": REPLY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    req = urllib.request.Request(
        RAMER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts-per-sub", type=int, default=25)
    parser.add_argument("--max-leads", type=int, default=15)
    parser.add_argument("--out", default="tools/leads.md")
    parser.add_argument("--no-draft", action="store_true", help="skip LLM draft, just list matches")
    args = parser.parse_args()

    print("[reddit-leads] getting OAuth token...")
    token = get_oauth_token()

    candidates = []
    for sub in SUBREDDITS:
        print(f"[reddit-leads] fetching r/{sub} hot posts...")
        try:
            posts = fetch_listing(sub, args.posts_per_sub, token)
        except urllib.error.HTTPError as e:
            print(f"[reddit-leads] r/{sub} listing failed: {e}")
            continue
        time.sleep(1.5)

        for post in posts:
            post_id = post["id"]
            post_title = post["title"]
            try:
                comments = fetch_comments(sub, post_id, token)
            except urllib.error.HTTPError as e:
                print(f"[reddit-leads]   post {post_id} comments failed: {e}")
                time.sleep(1.5)
                continue
            time.sleep(1.5)

            for c in comments:
                hits = score_text(c.get("body", ""))
                if hits:
                    candidates.append({
                        "subreddit": sub,
                        "post_title": post_title,
                        "post_permalink": f"https://reddit.com{post.get('permalink', '')}",
                        "comment_body": c["body"],
                        "comment_permalink": f"https://reddit.com{c.get('permalink', post.get('permalink', ''))}",
                        "author": c.get("author", "unknown"),
                        "hits": hits,
                    })

    candidates.sort(key=lambda c: len(c["hits"]), reverse=True)
    candidates = candidates[: args.max_leads]

    print(f"[reddit-leads] {len(candidates)} candidate comments found, drafting replies...")

    lines = ["# Reddit leads for Ramer\n", f"{len(candidates)} candidates, ranked by keyword match.\n"]
    for i, c in enumerate(candidates, 1):
        lines.append(f"\n## {i}. r/{c['subreddit']} -- u/{c['author']} (matched: {', '.join(c['hits'])})\n")
        lines.append(f"post: [{c['post_title']}]({c['post_permalink']})\n")
        lines.append(f"comment: [{c['comment_permalink']}]({c['comment_permalink']})\n")
        lines.append(f"\n> {c['comment_body'][:500]}\n")
        if not args.no_draft:
            try:
                draft = draft_reply(c["post_title"], c["comment_body"])
            except (urllib.error.URLError, KeyError) as e:
                draft = f"(draft failed -- is `python -m app.cli serve` running? error: {e})"
            lines.append(f"\n**draft reply:**\n\n{draft}\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[reddit-leads] wrote {args.out} -- review and post by hand, nothing was sent.")


if __name__ == "__main__":
    main()
