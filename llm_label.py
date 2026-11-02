"""
LLM pre-labeling for TakeMeter.

Run these in order (after find_ads.py has produced labels.csv):

  python llm_label.py label     LLM labels every post by its content -> llm_labels.csv
                                (safe to stop and rerun; it skips posts already done)
  python llm_label.py compare   Compares LLM labels to source labels -> review.csv
                                (only the posts where they disagree)
  python llm_label.py merge     After you fill the "final" column in review.csv,
                                writes the finished dataset -> labels_final.csv

Needs: pip install groq python-dotenv pandas
       and a .env file in this folder containing GROQ_API_KEY=your_key
"""
import html
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

INPUT_FILE = "labels.csv"          # dataset after ads were removed
LLM_FILE = "llm_labels.csv"
REVIEW_FILE = "review.csv"
FINAL_FILE = "labels_final.csv"

LABELS = ["rumor", "news", "opinion"]
MODEL = "openai/gpt-oss-120b"  # reasoning model: do not set max_tokens

# Mia's label definitions and decision rules
SYSTEM_PROMPT = """You label posts from celebrity and entertainment media.
Judge ONLY the content of the post, not which website it might come from.

rumor: An unverified claim about a celebrity, including speculation, blind items,
unnamed sourcing ("insiders say", "sources close to"), or "I heard" style gossip.
The claim is not backed by a traceable, named source.

news: A confirmed event, official statement, or verifiable fact about a celebrity,
attributed to a named source, publicist, the celebrity themself, or a documented
record (court filings, death certificates, official announcements, photos of a
public appearance).

opinion: The author's personal take, judgment, ranking, recommendation, review,
or argument about a celebrity or entertainment topic. The post is mainly the
author's perspective, not a report of an event.

Decision rules:
- If the verification relies on unnamed sources, label rumor, even if it includes
  specific details.
- If the main content is a reported fact about what a celebrity did or said, label
  news, even if the headline is dramatic or casual.
- If the main content is the author's recommendation, ranking, roundup of favorite
  looks, or judgment, label opinion.

Respond with exactly one word: rumor, news, or opinion."""


def get_client():
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY not found. Put it in a .env file in this folder.")
    from groq import Groq
    return Groq(api_key=key)


def ask_llm(client, text):
    """Return one of LABELS, or 'unparsed' if the reply can't be matched."""
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Post:\n{html.unescape(text)}"},
                ],
                temperature=0,
            )
            reply = (resp.choices[0].message.content or "").strip().lower()
            for label in LABELS:
                if label in reply:
                    return label
            return "unparsed"
        except Exception as e:
            if "model_not_found" in str(e) or "401" in str(e):
                sys.exit(f"Stopping: {e}")
            wait = 10 * (attempt + 1)
            print(f"  API error ({e}). Waiting {wait}s and retrying...")
            time.sleep(wait)
    return "unparsed"


def label():
    df = pd.read_csv(INPUT_FILE)
    df["row_id"] = df.index

    # Resume: skip posts already labeled in a previous run
    if os.path.exists(LLM_FILE):
        done = pd.read_csv(LLM_FILE)
    else:
        done = pd.DataFrame(columns=["row_id", "llm_label"])
    done_ids = set(done["row_id"])

    client = get_client()
    todo = df[~df["row_id"].isin(done_ids)]
    print(f"{len(done_ids)} already labeled, {len(todo)} to go.")

    rows = done.to_dict("records")
    for i, (_, r) in enumerate(todo.iterrows(), 1):
        rows.append({"row_id": r["row_id"], "llm_label": ask_llm(client, r["text"])})
        if i % 10 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(LLM_FILE, index=False)  # save progress
            print(f"  {i}/{len(todo)} done")
        time.sleep(2.1)  # stay under Groq free-tier rate limit

    print(f"Saved {LLM_FILE}")


def compare():
    df = pd.read_csv(INPUT_FILE)
    df["row_id"] = df.index
    llm = pd.read_csv(LLM_FILE)
    df = df.merge(llm, on="row_id", how="left")

    if df["llm_label"].isna().any():
        sys.exit("Some posts have no LLM label yet. Run: python llm_label.py label")

    if "note" not in df.columns:
        df["note"] = ""
    is_cold = df["note"].fillna("").str.strip().str.lower() == "cold"

    # Cold posts keep Mia's own label, so they are left out of the comparison
    check = df[~is_cold]
    agree = check["label"] == check["llm_label"]

    print(f"Compared {len(check)} posts (skipped {is_cold.sum()} cold posts).")
    print(f"Agree: {agree.sum()}  Disagree: {(~agree).sum()}  "
          f"({agree.mean():.0%} agreement)")
    print("\nSource label vs LLM label:")
    print(pd.crosstab(check["label"], check["llm_label"],
                      rownames=["source"], colnames=["llm"]))

    review = check[~agree][["row_id", "label", "llm_label", "text"]].copy()
    review = review.rename(columns={"label": "source_label"})
    review["final"] = ""
    review.to_csv(REVIEW_FILE, index=False)
    print(f"\nWrote {len(review)} posts to {REVIEW_FILE}. "
          f"Fill the 'final' column with rumor, news, opinion, or remove.")


def merge():
    df = pd.read_csv(INPUT_FILE)
    df["row_id"] = df.index
    llm = pd.read_csv(LLM_FILE)
    review = pd.read_csv(REVIEW_FILE)

    final = review["final"].fillna("").str.strip().str.lower()
    bad = ~final.isin(LABELS + ["remove"])
    if bad.any():
        sys.exit(f"{bad.sum()} rows in {REVIEW_FILE} have an empty or invalid 'final'. "
                 f"Use only: {', '.join(LABELS)}, or remove")
    review["final"] = final
    remove_ids = review.loc[review["final"] == "remove", "row_id"]
    review = review[review["final"] != "remove"]

    df = df.merge(llm, on="row_id", how="left")
    if "note" not in df.columns:
        df["note"] = ""
    df["note"] = df["note"].fillna("")
    is_cold = df["note"].str.strip().str.lower() == "cold"

    # Agreed posts: source and LLM matched
    df.loc[~is_cold, "note"] = "source_llm_agree"

    # Disagreed posts: use Mia's reviewed label
    decided = dict(zip(review["row_id"], review["final"]))
    for row_id, lab in decided.items():
        changed = lab != df.loc[row_id, "label"]
        df.loc[row_id, "label"] = lab
        df.loc[row_id, "note"] = "reviewed_changed" if changed else "reviewed_kept"

    df = df.drop(index=remove_ids)  # off-topic posts
    df[["text", "label", "note"]].to_csv(FINAL_FILE, index=False)

    print(f"Removed {len(remove_ids)} off-topic posts.")
    print(f"Saved {FINAL_FILE} with {len(df)} posts.")
    print("\nNotes:")
    print(df["note"].value_counts())
    print("\nFinal label counts:")
    counts = df["label"].value_counts()
    print(counts)
    top = counts.max() / counts.sum()
    print(f"\nLargest label is {top:.0%} of the data" +
          (" (over 70%, collect more of the thin labels)" if top > 0.70 else ""))


if __name__ == "__main__":
    steps = {"label": label, "compare": compare, "merge": merge}
    if len(sys.argv) != 2 or sys.argv[1] not in steps:
        print("Usage: python llm_label.py label | compare | merge")
    else:
        steps[sys.argv[1]]()