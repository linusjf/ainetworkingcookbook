#!/usr/bin/env python
"""
Returnjsonformat.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : returnjsonformat
# @created     : Monday Mar 16, 2026 16:23:10 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Simple base scenario
base_scenario = """
I have a network with the following devices:
- 2 core switches (Cisco Catalyst 9400)
- 4 distribution switches (Cisco Catalyst 9300)
- 8 access switches (Cisco Catalyst 9200)
Create an inventory document for this network.
"""
# Request JSON format
json_prompt = base_scenario + """
Return the inventory in JSON format with the following structure:
- Each device should have: hostname, model, role, mgmt_ip
- Group devices by role (core, distribution, access)
- Include a summary section with total counts
Provide ONLY the JSON output, no additional text.
"""
response_json = client.chat.completions.create(
model="gpt-4",
messages=[{"role": "user", "content": json_prompt}]
)
print("JSON Format Response:")
print(response_json.choices[0].message.content)
