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
import argparse
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_prompt(validate: bool = False) -> str:
    """Generate the prompt with optional JSON validation instructions."""
    base_scenario = """
I have a network with the following devices:
- 2 core switches (Cisco Catalyst 9400)
- 4 distribution switches (Cisco Catalyst 9300)
- 8 access switches (Cisco Catalyst 9200)
Create an inventory document for this network.
"""
    
    if validate:
        json_prompt = base_scenario + """
Return as valid JSON that passes the following validation:
1. All strings must be properly quoted
2. No trailing commas
3. Include a "validation" field with timestamp
4. Test that json.loads() can parse it successfully
After generating the JSON, mentally validate it against these rules before
returning.

Return the inventory in JSON format with the following structure:
- Each device should have: hostname, model, role, mgmt_ip
- Group devices by role (core, distribution, access)
- Include a summary section with total counts
Provide ONLY the JSON output, no additional text.
"""
    else:
        json_prompt = base_scenario + """
Return the inventory in JSON format with the following structure:
- Each device should have: hostname, model, role, mgmt_ip
- Group devices by role (core, distribution, access)
- Include a summary section with total counts
Provide ONLY the JSON output, no additional text.
"""
    
    return json_prompt

def validate_json(json_string: str) -> bool:
    """Validate that the JSON string can be parsed successfully."""
    try:
        parsed = json.loads(json_string)
        print("✓ JSON validation successful")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ JSON validation failed: {e}")
        return False

def main():
    """Main function to generate JSON inventory with optional validation."""
    parser = argparse.ArgumentParser(
        description="Generate network inventory in JSON format with optional validation."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Add JSON validation instructions and validate the output."
    )
    
    args = parser.parse_args()
    
    prompt = generate_prompt(args.validate)
    
    print("Generating JSON inventory...")
    if args.validate:
        print("(With JSON validation instructions)")
    
    response_json = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    json_content = response_json.choices[0].message.content
    
    print("\n" + "="*60)
    print("JSON Format Response:")
    print("="*60)
    print(json_content)
    
    if args.validate:
        print("\n" + "="*60)
        print("Validating JSON...")
        print("="*60)
        is_valid = validate_json(json_content)
        
        if is_valid:
            # Also check for validation field with timestamp
            parsed = json.loads(json_content)
            if "validation" in parsed:
                print(f"✓ Validation field found: {parsed['validation']}")
            else:
                print("✗ Validation field not found in JSON")
        else:
            print("✗ JSON failed validation")

if __name__ == "__main__":
    main()
