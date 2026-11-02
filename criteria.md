Criteria

1. Every label has an F1 of at least 0.60 on the test set. I picked 0.60 because news failed completely last time, and I want proof every label is actually learned by the AI model.
2. My worst label’s F1 is at least 70% of my best label’s F1. This protects against one label doing much worse than the others, like last time when news scored 0 while rumor scored 0.86.
3. I give the same label on at least 15 of 20 posts when I relabel my 20 cold posts without looking at my first labels. Since some posts sit on purpose between two labels, agreeing 75% of the time based on my label definition is reasonable.
4. Macro F1 beats the TF-IDF baseline by at least 0.1. If DistilBERT can’t beat a model that only counts words, then fine-tuning failed. I expect real fine-tuning to be much better than word counting.
5. Accuracy varies by no more than 0.05 across 3 seeds. I expect a stable result to trust that this is enough data to prove the AI model learned the labels.