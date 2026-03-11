#!/usr/bin/env python
"""
Helloworld.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : helloworld
# @created     : Wednesday Mar 11, 2026 11:47:03 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(model="gpt-4o",
                                            messages=[
                                                {
                                                    "role": "developer",
                                                    "content": "You are an AI assistant specialized in network engineering.."
                                                },
                                                {

                                                    "role": "user",
                                                    "content": "How do I install WiFi 7?",
                                                },
                                            ],
                                            )
print(completion.choices[0].message.content)
