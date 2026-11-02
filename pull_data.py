import csv
import time
import re
import requests
import xml.etree.ElementTree as ET

# Label assigned by SOURCE TYPE — gossip sites = rumor, magazines = news, commentary blogs = opinion
SOURCES = {
    "news": [
        "https://www.usmagazine.com/feed/",
        "https://www.eonline.com/syndication/feeds/rssfeeds/topstories.xml",
        "https://www.justjared.com/feed/",
        "https://www.hollywoodreporter.com/feed/",
        "https://variety.com/feed/",
    ],
    "rumor": [
        "https://pagesix.com/feed/",
        "https://www.tmz.com/rss.xml",
        "https://hollywoodlife.com/feed/",
        "https://www.thesun.co.uk/tvandshowbiz/feed/",
        "https://www.mirror.co.uk/3am/feed/",
        "https://www.heatworld.com/feed/",
        "https://okmagazine.com/feed/",
    ],
    "opinion": [
        "https://www.celebitchy.com/feed/",
        "https://dlisted.com/feed/",
        "https://laineygoSSip.com/feed/",
        "https://www.avclub.com/rss",
        "https://pitchfork.com/rss/news/feed.json",
        "https://consequence.net/feed/",
        "https://uproxx.com/feed/",
        "https://www.themarysue.com/feed/",
        "https://screenrant.com/feed/",
        "https://collider.com/feed/",
    ],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def clean(text):
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)       # strip HTML tags
    text = re.sub(r'&[a-z]+;', ' ', text)      # strip HTML entities
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_rss(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  {url} -> {r.status_code}")
        if r.status_code != 200:
            return []
        root = ET.fromstring(r.content)
        ns = {'content': 'http://purl.org/rss/1.0/modules/content/'}
        items = []
        for item in root.iter('item'):
            title = clean(item.findtext('title') or '')
            desc  = clean(item.findtext('description') or '')
            # some feeds put full text in content:encoded
            encoded = item.find('content:encoded', ns)
            body = clean(encoded.text if encoded is not None else '')
            # prefer description or body; title alone is too short
            text = body if len(body) > len(desc) else desc
            if not text:
                text = desc
            combined = f"{title}. {text}".strip() if title else text
            if len(combined) > 40:
                items.append(combined[:600])  # cap length
        return items
    except Exception as e:
        print(f"  ERROR {url}: {e}")
        return []

all_rows = []

for label, feeds in SOURCES.items():
    print(f"\nFetching '{label}' posts...")
    label_posts = []
    for feed_url in feeds:
        posts = fetch_rss(feed_url)
        label_posts.extend(posts)
        print(f"    got {len(posts)} items (running total: {len(label_posts)})")
        time.sleep(1)
    # deduplicate
    seen = set()
    unique = []
    for p in label_posts:
        key = p[:80]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    print(f"  Unique posts for '{label}': {len(unique)}")
    for post in unique:
        all_rows.append({"text": post, "label": label})

with open("deuxmoi_labeled.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["text", "label"])
    writer.writeheader()
    writer.writerows(all_rows)

from collections import Counter
counts = Counter(r["label"] for r in all_rows)
print(f"\nSaved {len(all_rows)} posts to deuxmoi_labeled.csv")
print("Label distribution:")
for label, count in counts.items():
    print(f"  {label}: {count}")