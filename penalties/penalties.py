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
import argparse
from typing import Optional


def generate_text(prompt: str, presence_penalty: float = 0.0, frequency_penalty: float = 0.0) -> str:
    """Generate text using OpenAI's API with specified penalty settings.
    
    Args:
        prompt: The user prompt to send to the model
        presence_penalty: Penalty for new tokens based on their presence in the text so far
        frequency_penalty: Penalty for new tokens based on their frequency in the text so far
    
    Returns:
        The generated text response from the model
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )
    return response.choices[0].message.content


def main() -> None:
    """Main function to run the penalties demonstration."""
    parser = argparse.ArgumentParser(description="Generate text with different penalty settings.")
    parser.add_argument("--prompt", type=str, default="Tell me a short story about a cat.",
                       help="Prompt to send to the model (default: 'Tell me a short story about a cat.')")
    
    args: argparse.Namespace = parser.parse_args()
    prompt: str = args.prompt
    
    # No penalties
    print("No penalties:\n", generate_text(prompt))
    
    # High presence penalty
    print("\nHigh Presence Penalty (2.0):\n", generate_text(prompt, presence_penalty=2.0))
    
    # High frequency penalty
    print("\nHigh Frequency Penalty (2.0):\n", generate_text(prompt, frequency_penalty=2.0))
    
    # Optimal penalties
    print("\nPresence Penalty (0.6),Frequency Penalty (0.5):\n", 
          generate_text(prompt, presence_penalty=0.6, frequency_penalty=0.5))


if __name__ == "__main__":
    main()
