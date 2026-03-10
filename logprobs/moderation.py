#!/usr/bin/env python
"""
Moderation.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : moderation
# @created     : Tuesday Mar 10, 2026 12:43:14 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import math
from openai import OpenAI

client = OpenAI()

labels = ["safe", "violence", "sexual", "hate"]

prompt = f"""
Classify the following text into one label:
{", ".join(labels)}

Text: TEXT
Label:
"""

text = [
"I will beat him until he can't stand.",
"He threatened to stab the man during the argument.",
"Those people are disgusting and should not live here.",
"The story contains explicit adult scenes between the characters.",
"The report discusses agricultural production trends in India.",
"He kept shouting insults. Later he said he would punch anyone who came closer.",
"I'm going to destroy that idiot when I see him.",
"The suspect allegedly assaulted two victims during the robbery.",
"This horror movie contains several brutal murder scenes."
]

for text_to_moderate in text:
    print()
    print(text_to_moderate)
    print()
    response = client.chat.completions.create(
        model="gpt-4.1",
            messages=[{"role": "user", "content": prompt.replace("TEXT",text_to_moderate)}],
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=10
        )

    if response.choices[0].logprobs is not None and response.choices[0].logprobs.content is not None:
        token_info = response.choices[0].logprobs.content[0].top_logprobs

        scores = {}

        for item in token_info:
            token = item.token.strip().lower()
            if token in labels:
                scores[token] = math.exp(item.logprob)

        print(scores)
        total = sum(scores.values())
        probs = {k: v / total for k, v in scores.items()}

        print("Category probabilities:")
        for k, v in probs.items():
            print(f"{k}: {v:.3f}")

        prediction = max(probs.items(), key=lambda x: x[1])[0]
        print("\nPredicted label:", prediction)
        print()
