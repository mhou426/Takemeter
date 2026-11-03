# TakeMeter: Celebrity News Classifier

A fine-tuned DistilBERT model that reads celebrity media posts and classifies each one as **rumor**, **news**, or **opinion**. The question behind it: can a model tell a claim backed by a named source from one built on unnamed sources or speculation?

**Version 2 results** (averaged over 3 random data splits):

| Label | F1 |
|---|---|
| opinion | 0.92 |
| news | 0.82 |
| rumor | 0.44 |
| **macro F1** | **0.73** |

Version 1 scored 0.52 macro F1 and 0 on news. Most of the improvement came from fixing the data, not the model.

The model detects **sourcing language**, not truth. It cannot fact check a claim; it learns whether a claim is attributed to a named source or document, or to unnamed sources and speculation.

---

## Labels

| Label | Definition | Example |
|---|---|---|
| rumor | An unverified claim about a celebrity: speculation, blind items, or unnamed sourcing ("insiders say," "a source close to") | "John Mayer and Kat Stickler quietly split a few months ago, according to a source." |
| news | A confirmed event or statement attributed to a named source, the celebrity themself, or a documented record (court filings, official announcements, their own posts) | "Liam Payne's estate will go to his son, court documents reveal." |
| opinion | The author's own take, review, ranking, recommendation, or commentary | A gossip blog post analyzing a royal feud in the author's voice |

**Decision rules for the hard boundaries:**

- Unnamed sources make it **rumor**, even if the post has specific details.
- A reported fact about what a celebrity did or said is **news**, even with a dramatic headline.
- If the main content is the author's recommendation, ranking, roundup, or judgment, it is **opinion**.

---

## Data

Posts come from public RSS feeds of celebrity and entertainment sites (tabloids, entertainment magazines, music and film sites, gossip blogs), collected with `pull_data.py`.

**Version 1 problem:** labels were assigned by which website a post came from. Many posts did not match their site's label, and the "news" class was full of shopping ads.

**Version 2 cleanup pipeline:**

1. **Removed ads and off-topic posts.** `find_ads.py` flags posts containing shopping words; each flagged post was checked by hand. A later pass removed posts unrelated to celebrities or entertainment (politics, general news, viral stories about non-celebrities, sponsored brand content). **65 of 337 posts removed (19%).**
2. **Relabeled by content.** `llm_label.py` had an open LLM (GPT OSS 120B via Groq) label every post using the definitions above, then compared its labels to the original source labels.
   - Agreed on 133 posts: label accepted.
   - Disagreed on 119 posts: reviewed, 83 changed and 36 kept.
3. **20 posts labeled by hand with no AI help** (marked `cold` in the `note` column), as an independent check.

**Final dataset:** `labels_final.csv`, 272 posts.

| Label | Count | Share |
|---|---|---|
| news | 140 | 51% |
| opinion | 89 | 33% |
| rumor | 43 | 16% |

The relabeling showed that tabloid "rumor" sites mostly publish documented facts (court records, bodycam footage, the celebrity's own posts), and music blogs labeled "opinion" mostly publish plain announcements. Real rumors with unnamed sources turned out to be rare.

---

## Training

Notebook: `takemeter_v2.ipynb` (Google Colab, T4 GPU).

| Setting | Value |
|---|---|
| Base model | distilbert-base-uncased (Hugging Face) |
| Split | 70 / 15 / 15 train / validation / test, stratified |
| Epochs | 10 |
| Learning rate | 5e-5 |
| Batch size | 16 |
| Warmup | 10% of training steps |
| Best epoch chosen by | validation macro F1 |
| Seeds | 42, 7, 2026 |

**The underfitting bug.** With 190 training posts and batch size 16, three epochs is only 36 weight updates. Version 1 also used 50 warmup steps, so the learning rate never reached its full value. The first version 2 run (3 epochs, 2e-5) predicted "news" for every post in two of three seeds. Raising epochs to 10 and the learning rate to 5e-5 fixed it. Keeping the epoch with the best validation macro F1 guards against overfitting from the extra epochs.

---

## Evaluation

Each seed creates a new random split. The baseline is TF-IDF plus logistic regression in scikit-learn, trained and tested on the same splits.

| Seed | Accuracy | Macro F1 | Baseline macro F1 | rumor F1 | news F1 | opinion F1 |
|---|---|---|---|---|---|---|
| 42 | 0.78 | 0.70 | 0.71 | 0.36 | 0.80 | 0.92 |
| 7 | 0.85 | 0.79 | 0.79 | 0.55 | 0.89 | 0.92 |
| 2026 | 0.76 | 0.70 | 0.63 | 0.40 | 0.78 | 0.92 |

**Confusion matrix, seed 2026** (rows are true labels, columns are predictions):

| true \ predicted | rumor | news | opinion |
|---|---|---|---|
| **rumor** | 3 | 3 | 0 |
| **news** | 5 | 16 | 0 |
| **opinion** | 1 | 1 | 12 |

### Success criteria

Set in `criteria.md` before training.

| Criterion | Target | Result | Verdict |
|---|---|---|---|
| 1. Every label F1 | at least 0.60 | news 0.78 to 0.89 and opinion 0.92 passed; rumor 0.36 to 0.55 failed | Missed (rumor only) |
| 2. Worst label F1 / best label F1 | at least 0.70 | 0.39 to 0.59 (rumor vs opinion) | Missed (rumor only) |
| 3. Agreement when relabeling the 20 cold posts | at least 15 of 20 | not yet run | Pending |
| 4. Macro F1 lead over baseline | at least 0.10 | -0.02 to +0.07 | Missed |
| 5. Accuracy spread across seeds | at most 0.05 | 0.095, under the common 0.10 stability line | Missed |

News and opinion pass in every seed. Criteria 1 and 2 miss only because of rumor, which has just 43 examples, and the small dataset also limits criteria 4 and 5. The next step is adding more rumor data.

**Why each miss happened:**

- **1 and 2: rumor is thin.** 43 posts total, 6 or 7 per test set, so one post changes its F1 a lot. News and opinion pass easily.
- **4: the baseline reads sourcing words well.** The labels hinge on phrases like "a source says" versus "court documents," which word counting picks up. With 190 training posts, DistilBERT has little room to add more.
- **5: small test set.** With 41 test posts, each post is worth 2.4 accuracy points; a spread of 0.095 is about 4 posts.

---

## Error analysis

8 of the 10 errors in seed 2026 are rumor versus news. They fall into two groups:

- **The model followed writing style instead of sourcing.** "John Mayer and Kat Stickler split, according to a source" was predicted news with 0.95 confidence; it comes from a news-style outlet. Liam Payne's estate ("court documents reveal") was predicted rumor; it is written in tabloid style. The model partly learned how an outlet writes, a leftover effect of collecting from fixed sources.
- **The label was wrong.** "Nick Cannon honors late son... posted a tribute" is labeled rumor, but by the rules above it is news, and the model said news.

**Hard cases while labeling:**

- Gossip blogs repeating a tabloid claim (royal family stories): opinion if the post is mainly the author's commentary, rumor if it just passes the claim on.
- "Everything to know" explainers: news if they report facts, opinion if they rank or recommend.
- A courtier quoted by name in a gossip blog: the quote is sourced, but the post is framed as commentary.

---

## Next steps

- Add 50 or more rumor posts with unnamed sourcing, from news-style outlets, to break the link between writing style and label.
- Run the cold relabel check (criterion 3).
- Try a larger model once the dataset grows.

---

## Repository

| File | Purpose |
|---|---|
| `pull_data.py` | Collects posts from RSS feeds |
| `find_ads.py` | Flags shopping posts for review and removes the confirmed ones |
| `llm_label.py` | LLM relabeling, comparison with source labels, and merging reviewed labels |
| `labels_final.csv` | Version 2 dataset (`text`, `label`, `note`) |
| `criteria.md` | Success criteria, written before training |
| `takemeter_v2.ipynb` | Training, baseline, 3-seed evaluation, criteria check |
| `results_three_seeds.json` | Version 2 results |
| `v1/` | Version 1 data, results, and planning notes |

## AI usage

- Claude assisted in writing the scripts, edited the training notebook, and helped analyze errors.
- GPT OSS 120B (via Groq) pre-labeled posts by content. Conflicting labels got a second AI review; I decided the uncertain cases, overriding the AI on 19 of 31.
