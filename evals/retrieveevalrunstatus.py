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

    run = client.evals.runs.retrieve(args.run_id, eval_id=args.eval_id)

    # User-friendly console output
    print("\n" + "="*60)
    print("EVALUATION RUN STATUS")
    print("="*60)
    print(f"Evaluation ID: {run.eval_id}")
    print(f"Run ID: {run.id}")
    print(f"Status: {run.status}")
    print(f"Created at: {run.created_at}")

    if hasattr(run, 'name') and run.name:
        print(f"Run name: {run.name}")

    if hasattr(run, 'report_url') and run.report_url:
        print(f"Report URL: {run.report_url}")

    # Show results summary if available
    if hasattr(run, 'result_counts'):
        if run.result_counts:
            print("\nResults Count:")
            for key, value in run.result_counts:
                print(f"  {key}: {value}")

    # Show error if failed
    if run.status == "failed" and hasattr(run, 'error'):
        if run.error:
            print(f"\nError: {run.error}")

    print("="*60 + "\n")

if __name__ == "__main__":
    main()
