#!/usr/bin/env python
"""
Querygemma.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : querygemma
# @created     : Wednesday Mar 11, 2026 16:09:54 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
from ollama import Client
client = Client(host='http://localhost:11434')
response = client.chat(model='gemma3:1b', messages=[
{
'role': 'user',
'content': 'What is Layer 2 of the OSI model?',
},
])
print(response['message']['content'])
