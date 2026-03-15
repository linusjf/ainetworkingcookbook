#!/usr/bin/env python
"""
Uploadtestdata.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : uploadtestdata
# @created     : Sunday Mar 15, 2026 18:38:55 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""

from openai import OpenAI
client = OpenAI()

file = client.files.create(
    file=open("tickets.jsonl", "rb"),
    purpose="evals"
)

print(file.id)
