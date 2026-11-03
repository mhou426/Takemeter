# TakeMeter : Planning Document

## Community

I chose celebrity gossip and entertainment media. The data comes from public RSS feeds across tabloids (Page Six, OK Magazine, The Sun), entertainment magazines (E! Online, US Magazine, Variety, Hollywood Reporter), and commentary blogs (Celebitchy, DListed, LaineyGossip, Screen Rant, Collider).

This community is a good fit for a classification task because the same subject matter (a celebrity's name, relationship, or career move) shows up in posts that range from pure speculation to confirmed reporting to editorial commentary. Readers in these spaces care about the difference between a verified story and gossip, and that distinction maps naturally to a labeling task.

## Labels

**rumor**: A post containing unverified claims about a celebrity, including speculation, blind items, unnamed sourcing ("insiders say," "sources close to"), or "I heard" style gossip. The defining feature is that the claim is not backed by a traceable, named source.

- Example 1: "Apparently a major pop star was spotted leaving a certain actor's apartment three nights in a row. My source works nearby and swears it's true."
- Example 2: "Word is that a certain A-list couple has been sleeping in separate bedrooms for months. No official comment yet but multiple people in their circle are talking."

**news**: A post reporting a confirmed event, official statement, or verifiable fact about a celebrity, attributed to a named source, publicist, or documented record. The claim is traceable and would hold up to basic fact-checking.

- Example 1: "Jennifer Lopez and Ben Affleck have officially filed for divorce. Both representatives released statements confirming the filing this morning."
- Example 2: "Nicki Minaj's $10 million lawsuit against an alleged stalker has been settled, per court records filed Tuesday."

**opinion**: A post expressing a personal take, critical judgment, theory, or argument about a celebrity or entertainment topic. The post is primarily the author's perspective, not a report of external events.

- Example 1: "Unpopular opinion but Taylor Swift's earlier country albums were genuinely better than anything she has released in the last five years."
- Example 2: "I genuinely think the backlash against Zendaya is completely manufactured. She has never done anything to deserve this level of criticism."

## Hard Edge Cases

The hardest boundary is between **news** and **rumor** when a post cites a specific detail (a name, a date, a number) but the source is unnamed or the publication is known for sensationalism. For example: "According to sources, Jennifer Aniston is reportedly engaged, and a friend close to the couple confirmed it to a tabloid." This has specificity (engagement, a confirming source) but the verification chain relies on unnamed sources and tabloid framing.

Decision rule: if the verification chain includes an unnamed source or a publication with a history of unverified reporting, label it **rumor**. A named publicist, court record, or official statement is required for **news**.

The second hard case is between **news** and **opinion** when a news article is written in an editorial voice, such as listicles, roundups, or product recommendations ("Miranda Kerr's favorite drugstore foundation"). Decision rule: if the primary content is a reported fact about a celebrity's actions or statements, label it **news** even if the framing is casual. If the primary content is the author's recommendation, ranking, or judgment, label it **opinion**.

## Data Collection Plan

Data will be collected from public RSS feeds grouped into three tiers by source type:

- Rumor feeds: Page Six, Hollywood Life, OK Magazine, The Sun (TV/showbiz)
- News feeds: US Magazine, E! Online, Just Jared, Hollywood Reporter, Variety
- Opinion feeds: Celebitchy, DListed, LaineyGossip, AV Club, Screen Rant, Collider, The Mary Sue

Labels are assigned by source domain: gossip tabloids map to rumor, entertainment magazines to news, commentary and criticism blogs to opinion. Target is 80+ examples per label. If any label falls under 70 after the first pull, additional feeds will be added for that category.

Post text is the concatenation of the RSS title and description fields, capped at 600 characters. HTML tags and entities are stripped. Duplicates are removed by comparing the first 80 characters.

## Evaluation Metrics

**Primary metric: per-class F1-score.** Accuracy alone is not enough because the dataset has mild class imbalance (opinion is 42%) and because the cost of misclassification is not uniform across labels. A model that predicts "opinion" for everything would get 42% accuracy but would be useless. Per-class F1 shows whether the model is learning each boundary or just defaulting to the majority class.

**Secondary metric: confusion matrix.** This shows the direction of errors, specifically which label pairs get confused and in which direction. The rumor/news boundary and the news/opinion boundary are expected to be the hardest cases, and the matrix reveals which one the model actually fails on.

**Definition of success:** The classifier is "good enough" if all three per-class F1 scores reach 0.65 or above. That threshold means the model is performing meaningfully above a random baseline (0.33) on every class, not just the majority class. A model with one class at F1 of 0.00 would not be deployable regardless of overall accuracy.

## AI Tool Plan

**Label stress-testing:** I will give Claude my label definitions and edge case descriptions and ask it to generate 5-10 posts that sit at the boundary between two labels. If it produces posts I cannot classify cleanly, that means the definitions need to be tighter before I commit to annotating 200+ examples.

**Annotation assistance:** Labels will be assigned programmatically by source domain rather than through example-by-example LLM pre-labeling. This approach is faster but introduces noise when a source's individual article does not match the expected register of its domain (e.g., an editorial product roundup on a "news" site). I will note this tradeoff in the evaluation.

**Failure pattern analysis:** After running the evaluation, I will paste the list of wrong predictions into Claude and ask it to identify common patterns (e.g., a specific label pair, post length, use of certain phrases) before writing the evaluation report. I will verify any patterns Claude surfaces by re-reading the examples myself.
