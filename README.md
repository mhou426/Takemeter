# TakeMeter: Celebrity Gossip Discourse Classifier

A fine-tuned text classifier that distinguishes between rumors, news, and opinion in celebrity gossip communities.

## Community

I chose celebrity gossip and entertainment media as the source community, drawing from public RSS feeds across tabloids (Page Six, OK Magazine, The Sun), entertainment magazines (E! Online, US Magazine, Variety, Hollywood Reporter), and commentary blogs (Celebitchy, DListed, LaineyGossip, Screen Rant, Collider). This community works well for a classification task because the same subject matter (a celebrity's name, relationship, career move) shows up in posts that range from unverified speculation to confirmed reporting to pure editorial commentary. That range is exactly what makes labeling interesting and nontrivial. Readers in gossip spaces genuinely care about whether a story is verified or not, and that distinction maps naturally to a classifier.

## Label Taxonomy

**rumor**: A post containing unverified claims about a celebrity, including speculation, blind items, unnamed sourcing ("insiders say," "sources close to"), or "I heard" style gossip. The defining feature is that the claim is not backed by a traceable, named source.

- Example 1: "Apparently a major pop star was spotted leaving a certain actor's apartment three nights in a row. My source works nearby and swears it's true."
- Example 2: "Word is that a certain A-list couple has been sleeping in separate bedrooms for months. No official comment yet but multiple people in their circle are talking."

**news**: A post reporting a confirmed event, official statement, or verifiable fact about a celebrity, attributed to a named source, publicist, or documented record. The claim is traceable and would hold up to basic fact-checking.

- Example 1: "Jennifer Lopez and Ben Affleck have officially filed for divorce. Both representatives released statements confirming the filing this morning."
- Example 2: "Nicki Minaj's $10 million lawsuit against an alleged stalker has been settled, per court records filed Tuesday."

**opinion**: A post expressing a personal take, critical judgment, theory, or argument about a celebrity or entertainment topic. The post is primarily the author's perspective, not a report of external events.

- Example 1: "Unpopular opinion but Taylor Swift's earlier country albums were genuinely better than anything she has released in the last five years."
- Example 2: "I genuinely think the backlash against Zendaya is completely manufactured. She has never done anything to deserve this level of criticism."

## Dataset

**Source:** Public RSS feeds from celebrity and entertainment media sites, collected June 2026.

- Rumor feeds: Page Six, Hollywood Life, OK Magazine, The Sun (TV/showbiz)
- News feeds: US Magazine, E! Online, Just Jared, Hollywood Reporter, Variety
- Opinion feeds: Celebitchy, DListed, LaineyGossip, AV Club, Screen Rant, Collider, The Mary Sue

**Labeling process:** Labels were assigned by source domain. Gossip tabloids mapped to `rumor`, entertainment magazines to `news`, and commentary/criticism blogs to `opinion`. For each post, the RSS title and description were concatenated into a single text field capped at 600 characters. HTML tags and entities were stripped. Duplicates were removed by comparing the first 80 characters.

This is a proxy labeling strategy: source type generally predicts content type, but individual articles may not match the domain default. I discuss the consequences of this in the Reflection section.

**Label distribution:**

| Label   | Count | Percentage |
|---------|-------|------------|
| opinion | 142   | 42.1%      |
| rumor   | 100   | 29.7%      |
| news    | 95    | 28.2%      |
| **Total** | **337** | |

No single label exceeds 42% of the dataset. The train/validation/test split was 70/15/15 (235 train, 51 validation, 51 test), stratified by label.

**Difficult-to-label examples:**

1. *"Hacks' Robby Hoffman Reveals She Stormed Out of 2025 Emmys After Loss."* The headline uses dramatic, emotionally charged language ("stormed out") that sounds like unverified gossip. But the content is a first-person account from Hoffman herself, making it a confirmed statement. I labeled it **news** because the source is the subject of the story, not an unnamed third party.

2. *"Miranda Kerr's 'Fresh, Natural' Drugstore Foundation Pick Is Only $9."* This is a product recommendation article from a celebrity news site. The celebrity is real and the product claim is verifiable, but the primary content is editorial recommendation rather than event reporting. I labeled it **news** because it came from E! Online (a news-coded source), but in retrospect the content reads more like opinion. This case exposed a real weakness in source-based labeling.

3. *"Was Taylor Parker Diagnosed With a Disorder? Inside Her Mental Health."* A longform article about a real person's verified legal case and mental health record, sourced from court documents. The question-framing headline sounds speculative, but the body cites documented records. I labeled it **news** despite the sensational headline because the sourcing is traceable.

## Fine-Tuning Pipeline

**Base model:** `distilbert-base-uncased` from HuggingFace

**Training platform:** Google Colab with free T4 GPU

**Training setup:**

| Parameter      | Value |
|----------------|-------|
| Epochs         | 3     |
| Learning rate  | 2e-5  |
| Train batch    | 16    |
| Eval batch     | 32    |
| Weight decay   | 0.01  |
| Warmup steps   | 50    |
| Best model by  | Validation accuracy |

**Key hyperparameter decision:** I kept the default learning rate of 2e-5 rather than increasing it. With only 235 training examples, a higher learning rate risks overshooting the loss minimum and overfitting to the small training set. The training loss decreased steadily across all three epochs (1.087 to 1.051 to 1.016) without diverging, which confirmed that 2e-5 was appropriate for this dataset size. Validation accuracy improved significantly only in epoch 3 (from 0.431 to 0.706), suggesting the model needed all three epochs to adapt its pretrained representations to this domain before the classification head could generalize.

## Baseline

**Approach:** Zero-shot classification using `llama-3.3-70b-versatile` via the Groq API. Each test example was passed to the model with a system prompt containing the three label definitions and one example post per label. The model was instructed to respond with only the label name. Temperature was set to 0 for deterministic output.

**Prompt used:**

```
You are classifying posts from a celebrity gossip community.
Assign each post to exactly one of the following categories.

rumor: Unverified claims, speculation, blind items, unnamed sources, "I heard" stories about celebrities.
Example: "Apparently a major pop star was spotted leaving a certain actor's apartment. My source works nearby and swears it's true."

news: Confirmed events, official statements, verifiable facts from named sources about celebrities.
Example: "Jennifer Lopez and Ben Affleck have officially filed for divorce. Their reps both released statements this morning."

opinion: Personal takes, judgments, theories, and views about celebrities. "I think," unpopular opinion, arguments for a position.
Example: "Unpopular opinion but Taylor Swift's earlier country albums were genuinely better than anything she's released recently."

Respond with ONLY the single word: rumor, news, or opinion
Do not explain your reasoning.
```

All 51 test responses were parseable (0 unparseable).

## Evaluation Report

### Results Summary

| Model                        | Accuracy |
|------------------------------|----------|
| Zero-shot baseline (Groq)    | 0.373    |
| Fine-tuned DistilBERT        | 0.647    |
| **Improvement**              | **+0.275** |

Test set size: 51 examples (15 rumor, 15 news, 21 opinion).

### Per-Class Metrics: Fine-Tuned Model

| Label   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| rumor   | 0.86      | 0.80   | 0.83 | 15      |
| news    | 0.00      | 0.00   | 0.00 | 15      |
| opinion | 0.57      | 1.00   | 0.72 | 21      |

### Per-Class Metrics: Baseline

| Label   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| rumor   | 0.25      | 0.07   | 0.11 | 15      |
| news    | 0.32      | 0.87   | 0.46 | 15      |
| opinion | 0.83      | 0.24   | 0.37 | 21      |

### Confusion Matrix: Fine-Tuned Model

|              | Predicted: rumor | Predicted: news | Predicted: opinion |
|--------------|------------------|-----------------|--------------------|
| True: rumor  | 12               | 0               | 3                  |
| True: news   | 2                | 0               | 13                 |
| True: opinion| 0                | 0               | 21                 |

The model never predicts `news`. All 15 true news examples were misclassified: 13 as opinion and 2 as rumor.

### Wrong Prediction Analysis

**Error 1: news predicted as opinion (confidence: 0.40)**

> "Top Gun Star James Handy's Death Certificate Reveals Sad New Details..."

True label: news. This article reports a verifiable fact (contents of a public death certificate). However, the headline uses emotionally charged language ("Sad New Details") that is indistinguishable in tone from an editorial take. The model learned to associate emotive framing with opinion, and this news article's presentation triggered that pattern. This is a data problem: many "news" articles in the training set were written in a sensational editorial voice, so the model had no reliable text signal to separate news from opinion.

**Error 2: news predicted as opinion (confidence: 0.38)**

> "Miranda Kerr's 'Fresh, Natural' Drugstore Foundation Pick Is Only $9..."

True label: news (by source domain). The content is a product recommendation, which is essentially editorial. This example exposed the core labeling flaw: the label was assigned based on the source (E! Online = news) rather than the content. A product recommendation from E! Online reads identically to opinion writing from Celebitchy. The model identified the content as opinion-like, which is arguably the correct reading of the text. The label was wrong by the source-based convention I used during data collection.

**Error 3: news predicted as rumor (confidence: 0.42)**

> "Hacks' Robby Hoffman Reveals She Stormed Out of 2025 Emmys After Loss..."

True label: news. This is a confirmed statement from the subject herself. However, "stormed out" is a tabloid-style phrase, and "reveals" is a word pattern common in gossip and rumor posts ("sources reveal," "insiders reveal"). The model associated this phrasing with the rumor class, which is a reasonable surface-level inference given the training data. This error points to the difficulty of separating news from rumor when news sources adopt tabloid language.

### Sample Classifications

| Post (truncated) | True | Predicted | Confidence |
|------------------|------|-----------|------------|
| The Strokes Follow A Big Coachella Weekend By Announcing A 2026 World Tour... | opinion | opinion | 0.63 |
| 'One Night Only': Everything To Know About Callum Turner And Monica Barbaro's Rom-Com... | opinion | opinion | 0.57 |
| It's The Series Finale. I cannot believe that after all these years, my time with Dlisted... | opinion | opinion | 0.62 |
| Nicki Minaj's $10 Million Legal Fight With Alleged 'Stalker' is Over... | rumor | rumor | 0.39 |
| Jennifer Lopez: Mom with the hookups. Jennifer Lopez was one of so many celebrities... | opinion | opinion | 0.57 |

**Correct prediction explained:** The Strokes tour announcement post was correctly labeled opinion (confidence 0.63). The post is written in an editorial voice typical of music commentary blogs, combining factual details (Coachella, 2026 world tour) with framing that signals authorial perspective ("a busy spring and summer" reads as commentary, not reporting). The model correctly associated this register with opinion. This is the kind of post where fine-tuning genuinely helped, since the baseline only achieved 0.24 recall on opinion.

## Reflection: What the Model Learned vs. What Was Intended

The intended behavior was a three-way classifier distinguishing verified facts (news) from unverified claims (rumor) from editorial perspective (opinion). What the model actually learned was a two-way classifier: it reliably separates rumor from opinion, but it collapsed news entirely into opinion.

The core failure is a labeling design problem, not a model problem. Labels were assigned by source domain. E! Online and Variety posts were tagged as `news` because of where they came from, not because of what they contained. But celebrity news articles are frequently written in an editorial style: listicles, product roundups, style retrospectives, emotionally framed headlines. These texts are linguistically indistinguishable from opinion writing. The model had no reliable text signal to separate the two classes because the training data never reflected a consistent text-level distinction between them.

The rumor class succeeded because tabloid writing has genuinely distinct surface features: unnamed sourcing language, present-tense speculation, and dramatic hedging ("reportedly," "insiders say") that appear consistently across examples. The opinion class succeeded for the same reason: editorial voice, subjective framing, and first-person takes are consistent markers. News failed because the boundary was defined by metadata (source domain) rather than by anything in the text itself.

If I were to improve this, the single highest-impact change would be a manual annotation pass on the 95 news examples, relabeling each one based on its actual content rather than its source. That would likely move a significant number of product roundups and editorial listicles from news to opinion, giving the model a cleaner signal and bringing news F1 above zero.

## Spec Reflection

**How the spec helped:** The spec's guidance on strong vs. weak taxonomies pushed me to write decision rules for edge cases (the rumor/news boundary, the news/opinion boundary) before starting data collection. That upfront investment made the labeling process faster and helped me recognize the domain-labeling tradeoff explicitly rather than discovering it only during evaluation.

**Where implementation diverged:** The spec assumed manual collection of real community posts, with the researcher reading and labeling each example individually. I attempted Reddit scraping first but was blocked by their API access restrictions. I pivoted to RSS feed collection with source-based labeling, which is faster and produces a larger dataset but removes the researcher from the annotation decision entirely. Labels get assigned by which website a post came from, not by reading the post. The result is a dataset with systematic label noise in the news category, and that noise is the primary driver of the model's failure on that class. A manual annotation pass on the news examples would likely have caught the product-recommendation mislabeling and improved news F1 substantially.

## AI Usage

**Instance 1: Data collection pipeline.** I directed Claude to write the RSS feed scraping script (`pull_data.py`) after Reddit API access was blocked. Claude produced multiple iterations of the script as individual feeds returned 403/404 errors and as a Windows console encoding issue broke all output. I executed each version, reported the output counts, and Claude adjusted the feed list to balance the label distribution. The final script and feed selection reflect several rounds of debugging against live feed responses.

**Instance 2: Documentation drafting.** I directed Claude to draft both planning.md and README.md using the actual model outputs, confusion matrix numbers, and wrong prediction text from my Colab notebook. I provided all numerical results and the full set of wrong and correct predictions as inputs. Claude produced draft text. I reviewed the drafts for accuracy against the notebook output and revised the analysis and tone to match my voice.

**Instance 3: Failure pattern analysis.** After running the evaluation in Section 4, I shared the 15 wrong predictions with Claude and asked it to identify common patterns. Claude identified the systematic news-to-opinion collapse and linked it to the source-based labeling strategy. I verified this by re-reading the misclassified examples and confirmed that the pattern held: most "news" errors were articles with editorial framing that happened to come from news-coded sources.

**Annotation assistance disclosure:** No LLM pre-labeling of individual posts was used. Labels were assigned programmatically by source domain. The source-to-label mapping (tabloid = rumor, magazine = news, blog = opinion) was designed collaboratively with Claude and applied in the scraping script.

## Demo Video

https://www.loom.com/share/c67deab514974af6aa76466e6a2ed560
