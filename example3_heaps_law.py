# EXAMPLE 3 - A simple script to illustrate Heaps' law

import math
import os


def simple_tokenize(text):
    text = text.lower()
    for ch in ",.;:!?()?\"'":
        text = text.replace(ch, " ")
    return [t for t in text.split() if t]


def growth_curve(tokens, step=1000):
    results = []
    seen = set()
    N = 0
    for i, w in enumerate(tokens, start=1):
        N += 1
        seen.add(w)
        if i % step == 0:
            V = len(seen)
            results.append((N, V))
    return results


file_name = "korpus.txt"
if not os.path.exists(file_name):
    print(f"Plik '{file_name}' nie istnieje — tworzę przykładowy korpus.")
    dummy_content = (
        "This is a sample text for the korpus. Lotem ipsum it dolor lolololo. "
        "It contains several words to test the tokenization and growth curve functions. "
        "Natural language processing is interesting. Machine learning is also interesting."
    )
    with open(file_name, "w", encoding="utf8") as f:
        f.write(dummy_content)

with open(file_name, "r", encoding="utf8") as f:
    full_text = f.read()

tokens = simple_tokenize(full_text)

curve = growth_curve(tokens, step=5)

print("N_tokens\tV_types\tlog10(N)\tlog10(V)")
for N, V in curve:
    print(f"{N}\t{V}\t{math.log10(N):.3f}\t{math.log10(V):.3f}")
