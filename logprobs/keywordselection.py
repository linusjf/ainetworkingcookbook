#!/usr/bin/env python
"""
Keywordselection.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : keywordselection
# @created     : Tuesday Mar 10, 2026 18:19:54 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import math
from typing import Dict, List
from openai import OpenAI

client = OpenAI()

text = """
PostgreSQL indexing improves query performance by allowing the database
engine to quickly locate rows without scanning the entire table.
Common index types include B-tree, Hash, and GiST indexes.
"""

prompt = f"""
Extract the most important keywords from the text.
Return a comma separated list of keywords.

Text:
{text}

Keywords:
"""

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=20,
    logprobs=True,
    top_logprobs=5
)

assert response.choices[0].logprobs is not None
tokens = response.choices[0].logprobs.content

assert tokens is not None

keywords: Dict[str, float] = {}

current_word: List[str] = []
current_score = 0.0

for t in tokens:
    token = t.token
    prob = math.exp(t.logprob)

    if token in [",", "\n"]:
        if current_word:
            word = "".join(current_word).lower()
            keywords[word] = keywords.get(word, 0.0) + current_score
        current_word = []
        current_score = 0.0
        continue

    current_word.append(token.strip())
    current_score += prob

if current_word:
    word = "".join(current_word).lower()
    keywords[word] = keywords.get(word, 0.0) + current_score

ranked = sorted(keywords.items(), key=lambda x: x[1], reverse=True)

for word, score in ranked:
    print(f"{word}: {score:.4f}")
