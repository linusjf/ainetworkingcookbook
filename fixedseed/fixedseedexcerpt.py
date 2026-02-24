#!/usr/bin/env python
"""
Fixedseedexcerpt.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : fixedseedexcerpt
# @created     : Tuesday Feb 24, 2026 14:23:26 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import openai
import asyncio
import textwrap
from IPython.display import display, HTML

from utils.embeddings_utils import (
    get_embedding,
    distances_from_embeddings
)

GPT_MODEL = "gpt-3.5-turbo-1106"

async def get_chat_response(
    system_message: str, user_request: str, seed: int = None, temperature: float = 0.7
):
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request},
        ]

        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            seed=seed,
            max_tokens=200,
            temperature=temperature,
        )

        response_content = response.choices[0].message.content
        system_fingerprint = response.system_fingerprint
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.total_tokens - response.usage.prompt_tokens

        rows = [
            ("Response", response_content),
            ("System Fingerprint", system_fingerprint),
            ("Prompt Tokens", prompt_tokens),
            ("Completion Tokens", completion_tokens),
        ]

        print(format_console_table(rows))

        return response_content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def format_console_table(rows, max_width=80):
    key_width = 20
    value_width = max_width - key_width - 5  # borders + spaces

    horizontal = f"+{'-'*(key_width+2)}+{'-'*(value_width+2)}+"
    lines = [horizontal]

    for key, value in rows:
        key = str(key)
        value = str(value)

        wrapped_value = textwrap.wrap(value, width=value_width) or [""]

        for i, chunk in enumerate(wrapped_value):
            if i == 0:
                key_cell = key.ljust(key_width)
            else:
                key_cell = " " * key_width

            lines.append(
                f"| {key_cell} | {chunk.ljust(value_width)} |"
            )

    lines.append(horizontal)
    return "\n".join(lines)


def calculate_average_distance(responses):
    """
    This function calculates the average distance between the embeddings of the responses.
    The distance between embeddings is a measure of how similar the responses are.
    """
    # Calculate embeddings for each response
    response_embeddings = [get_embedding(response) for response in responses]

    # Compute distances between the first response and the rest
    distances = distances_from_embeddings(response_embeddings[0], response_embeddings[1:])

    # Calculate the average distance
    average_distance = sum(distances) / len(distances)

    # Return the average distance
    return average_distance

async def main():
    topic = "a journey to Mars"
    system_message = "You are a helpful assistant."
    user_request = f"Generate a short excerpt of news about {topic}."

    responses = []

    async def get_response(i):
        print(f'Output {i + 1}\n{"-" * 10}')
        response = await get_chat_response(
            system_message=system_message, user_request=user_request
        )
        return response

    responses = await asyncio.gather(*[get_response(i) for i in range(5)])
    average_distance = calculate_average_distance(responses)
    print(f"The average similarity between responses is: {average_distance}")

    SEED = 123
    responses = []


    async def get_seeded_response(i):
        print(f'Output {i + 1}\n{"-" * 10}')
        response = await get_chat_response(
            system_message=system_message,
            seed=SEED,
            temperature=0,
            user_request=user_request,
        )
        return response


    responses = await asyncio.gather(*[get_seeded_response(i) for i in range(5)])

    average_distance = calculate_average_distance(responses)
    print(f"The average distance between responses is: {average_distance}")

if __name__ == "__main__":
    asyncio.run(main(), debug=False)
