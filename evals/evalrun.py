#!/usr/bin/env python
"""
Evalrun.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : evalrun
# @created     : Sunday Mar 15, 2026 18:42:50 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""

import argparse
from openai import OpenAI

def main():
    """Main function to run the eval with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an OpenAI evaluation with specified eval ID and file ID."
    )
    parser.add_argument(
        "eval_id",
        help="The evaluation ID to use for the run."
    )
    parser.add_argument(
        "file_id",
        help="The file ID containing the data for the evaluation."
    )
    parser.add_argument(
        "--name",
        default="Categorization text run",
        help="Optional name for the evaluation run. Default: 'Categorization text run'"
    )

    args = parser.parse_args()

    client = OpenAI()

    run = client.evals.runs.create(
        args.eval_id,
        name=args.name,
        data_source={
            "type": "responses",
            "model": "gpt-4.1",
            "input_messages": {
                "type": "template",
                "template": [
                    {"role": "developer", "content": "You are an expert in categorizing IT support tickets. Given the support ticket below, categorize the request into one of 'Hardware', 'Software', or 'Other'. Respond with only one of those words."},
                    {"role": "user", "content": "{{ item.ticket_text }}"},
                ],
            },
            "source": {"type": "file_id", "id": args.file_id},
        },
    )

    print(run.id)

if __name__ == "__main__":
    main()
