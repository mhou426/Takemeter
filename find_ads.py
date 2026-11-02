"""
Find and remove shopping/ad posts from the TakeMeter dataset.

Step 1 (find):    python find_ads.py find
    Writes shopping_candidates.csv with every post containing a shopping word.
    Open it and type yes or no in the "ad?" column for each row.

Step 2 (remove):  python find_ads.py remove
    Reads your reviewed shopping_candidates.csv and removes every row
    you marked yes from deuxmoi_labeled.csv. Saves the result as labels.csv.
"""
import sys
import pandas as pd

DATA_FILE = "Copy_of_deuxmoi_labeled.csv"
CANDIDATES_FILE = "shopping_candidates.csv"
OUTPUT_FILE = "labels.csv"

# Words that usually mean a post is selling something, not discussing a celebrity
SHOPPING_WORDS = [
    "sale", "deal", "deals", "prime day", "shop", "amazon", "discount",
    "% off", "coupon", "storefront", "skincare", "makeup", "lip balm",
    "lip oil", "foundation",
]


def find():
    df = pd.read_csv(DATA_FILE)
    df["row_id"] = df.index  # remember each post's original position

    text = df["text"].str.lower()
    is_candidate = text.apply(lambda t: any(word in t for word in SHOPPING_WORDS))
    # prices like "$9" or "$24"
    is_candidate |= text.str.contains(r"\$\d", regex=True)

    candidates = df[is_candidate][["row_id", "label", "text"]].copy()
    candidates["ad?"] = ""
    candidates.to_csv(CANDIDATES_FILE, index=False)
    print(f"Found {len(candidates)} candidates out of {len(df)} posts.")
    print(candidates["label"].value_counts())


def remove():
    df = pd.read_csv(DATA_FILE)
    reviewed = pd.read_csv(CANDIDATES_FILE)

    answers = reviewed["ad?"].fillna("").str.strip().str.lower()
    if (answers == "").any():
        print(f"{(answers == '').sum()} rows still have no yes/no. Finish reviewing first.")
        return

    ad_ids = reviewed.loc[answers == "yes", "row_id"]
    cleaned = df.drop(index=ad_ids)
    cleaned.to_csv(OUTPUT_FILE, index=False)

    print(f"Removed {len(ad_ids)} ads. {len(cleaned)} posts left, saved to {OUTPUT_FILE}.")
    print("Removed per label:")
    print(df.loc[ad_ids, "label"].value_counts())
    print("New label counts:")
    print(cleaned["label"].value_counts())


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("find", "remove"):
        print("Usage: python find_ads.py find   OR   python find_ads.py remove")
    elif sys.argv[1] == "find":
        find()
    else:
        remove()
