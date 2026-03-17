# EXAMPLE 1 - A simple bigram language model in Python

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


def sentence_log_prob(sentence, bigram_probs, vocab):
    toks = tokenize(sentence)
    log_p = 0.0
    for w1, w2 in zip(toks[:-1], toks[1:]):
        if w1 not in vocab:
            w1 = "<s>"
        if w2 not in vocab:
            w2 = "</s>"
        log_p += math.log2(bigram_probs[w1][w2])
    return log_p


def corpus_perplexity(texts, bigram_probs, vocab):
    log_prob_sum = 0.0
    token_count = 0
    for txt in texts:
        toks = tokenize(txt)
        token_count += len(toks) - 1
        log_prob_sum += sentence_log_prob(txt, bigram_probs, vocab)
    avg_log_prob = log_prob_sum / token_count
    return 2 ** (-avg_log_prob)


train_texts = [
    "I like natural language processing",
    "I like machine learning",
    "language models predict words"
]

test_texts = [
    "I like language models",
    "machine learning is fun",
    "I like fancy duba buba dada little dogs",
]

bigram_probs, unigram_counts, vocab = train_bigram_model(train_texts)

for s in test_texts:
    lp = sentence_log_prob(s, bigram_probs, vocab)
    print(f"Log2 P({s}) = {lp:.3f}")

pp = corpus_perplexity(test_texts, bigram_probs, vocab)
print(f"Perplexity on test set = {pp:.3f}")
