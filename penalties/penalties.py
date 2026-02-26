#!/usr/bin/env python
"""
Penalties.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : penalties
# @created     : Thursday Feb 26, 2026 18:59:51 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import openai

def generate_text(prompt, presence_penalty=0.0, frequency_penalty=0.0):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )
    return response.choices[0].message.content

# Example prompt
prompt = "Tell me a short story about a cat."

# No penalties
print("No penalties:\n", generate_text(prompt))

# High presence penalty
print("\nHigh Presence Penalty (2.0):\n", generate_text(prompt, presence_penalty=2.0))

# High frequency penalty
print("\nHigh Frequency Penalty (2.0):\n", generate_text(prompt, frequency_penalty=2.0))


# Optimal penalties
print("\nPresence Penalty (0.6),Frequency Penalty (0.5):\n", generate_text(prompt, presence_penalty=0.6, frequency_penalty=0.5))
