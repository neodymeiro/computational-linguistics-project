# EXAMPLE 2 - Computing word surprisal in a bigram model

import math
from collections import Counter, defaultdict


def tokenize(text):
    text = text.lower()
    tokens = text.replace("\n", " ").split()
    return ["<s>"] + tokens + ["</s>"]


def train_bigram_model(texts):
    unigram_counts = Counter()
    bigram_counts = Counter()

    for txt in texts:
        toks = tokenize(txt)
        unigram_counts.update(toks)
        bigram_counts.update(zip(toks[:-1], toks[1:]))

    vocab = set(unigram_counts.keys())
    V = len(vocab)

    bigram_probs = defaultdict(dict)
    for (w1, w2), c in bigram_counts.items():
        bigram_probs[w1][w2] = (c + 1) / (unigram_counts[w1] + V)

    for w1 in vocab:
        for w2 in vocab:
            if w2 not in bigram_probs[w1]:
                bigram_probs[w1][w2] = 1 / (unigram_counts[w1] + V)

    return bigram_probs, unigram_counts, vocab


def word_surprisals(sentence, bigram_probs, vocab):
    toks = tokenize(sentence)
    results = []
    for w1, w2 in zip(toks[:-1], toks[1:]):
        if w2 not in bigram_probs[w1]:
            w2_eff = "</s>" if w2 not in vocab else w2
        else:
            w2_eff = w2
        p = bigram_probs[w1][w2_eff]
        s = -math.log2(p)
        results.append((w2, p, s))
    return results


train_texts = [
    "I like natural language processing",
    "I like machine learning",
    "language models predict words"
]

bigram_probs, unigram_counts, vocab = train_bigram_model(train_texts)

example_sentence = "I like language models"
surps = word_surprisals(example_sentence, bigram_probs, vocab)

print("Word\tP(word | context)\tSurprisal")
for w, p, s in surps:
    print(f"{w}\t{p:.4f}\t\t{s:.3f}")
