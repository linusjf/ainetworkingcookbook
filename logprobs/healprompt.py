#!/usr/bin/env python
"""
Healprompt.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : healprompt
# @created     : Tuesday Mar 10, 2026 19:12:06 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import math
from openai import OpenAI

client = OpenAI()

def heal_prompt(prefix: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "Complete the unfinished word at the end of the text. Return only the missing characters."
            },
            {"role": "user", "content": prefix}
        ],
        max_tokens=1,
        temperature=0
    )

    token = response.choices[0].message.content
    assert token is not None
    return prefix + token

prompts = [
    "The theory of relat",
    "PostgreSQL inde",
    "Machine lear",
    "The capital of Fran",
    "Database optimi",
    "Artificial intelli",
    "The history of compu",
    "Neural netw",
]

for p in prompts:
    healed = heal_prompt(p)
    print("Original:", p)
    print("Healed:  ", healed)
    print()
