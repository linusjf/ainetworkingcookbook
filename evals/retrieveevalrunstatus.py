#!/usr/bin/env python
"""
Retrieveevalrunstatus.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : retrieveevalrunstatus
# @created     : Monday Mar 16, 2026 11:42:58 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""

import argparse
from openai import OpenAI

def main():
    """Main function to retrieve eval run status with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrieve the status of an OpenAI evaluation run."
    )
    parser.add_argument(
        "eval_id",
        help="The evaluation ID."
    )
    parser.add_argument(
        "run_id",
        help="The evaluation run ID."
    )
    
    args = parser.parse_args()
    
    client = OpenAI()
    
    run = client.evals.runs.retrieve(args.eval_id, args.run_id)
    print(run)

if __name__ == "__main__":
    main()
