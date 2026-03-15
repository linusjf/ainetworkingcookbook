#!/usr/bin/env python
"""
Classifyitsupporttickets.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : classifyitsupporttickets
# @created     : Sunday Mar 15, 2026 18:28:19 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""

from openai import OpenAI
client = OpenAI()

instructions = """
You are an expert in categorizing IT support tickets. Given the support
ticket below, categorize the request into one of "Hardware", "Software",
or "Other". Respond with only one of those words.
"""

ticket = "My monitor won't turn on - help!"

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "developer", "content": instructions},
        {"role": "user", "content": ticket},
    ],
)

print(response.output_text)
