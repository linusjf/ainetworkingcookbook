#!/usr/bin/env python3
"""
Counttokens.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : counttokens
# @created     : Wednesday Feb 25, 2026 15:13:22 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
import tiktoken

token_string = "tiktoken is great!"
encoding = tiktoken.get_encoding("cl100k_base")

print(encoding)

print(f"Encoding '{token_string}'")

encoded = encoding.encode(token_string)

print(encoded)

decoded_string = encoding.decode(encoded)
print(f"Decoded {encoded} to '{decoded_string}'")
print([encoding.decode_single_token_bytes(token) for token in encoded])

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

print(encoding)

print(f"Encoding '{token_string}'")
encoded = encoding.encode(token_string)
print(encoded)
decoded_string = encoding.decode(encoded)
print(f"Decoded {encoded} to '{decoded_string}'")
print([encoding.decode_single_token_bytes(token) for token in encoded])

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

encoding_name = "o200k_base"
print(f"Getting number of tokens for string '{token_string}' using encoding '{encoding_name}'")
token_count = num_tokens_from_string(token_string, encoding_name)
print(f"Token count is {token_count}")

def compare_encodings(example_string: str) -> None:
    """Prints a comparison of three string encodings."""
    # print the example string
    print(f'\nExample string: "{example_string}"')
    # for each encoding, print the # of tokens, the token integers, and the token bytes
    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        num_tokens = len(token_integers)
        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
        print()
        print(f"{encoding_name}: {num_tokens} tokens")
        print(f"token integers: {token_integers}")
        print(f"token bytes: {token_bytes}")

compare_encodings("antidisestablishmentarianism")
compare_encodings("2 + 2 = 4")
compare_encodings("お誕生日おめでとう")

def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# let's verify the function above matches the OpenAI API response

from openai import OpenAI
import os
from openai.types.chat import ChatCompletionMessageParam

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

example_messages:list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Things working well together will increase revenue.",
    },
    {
        "role": "system",
        "name": "example_user",
        "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
    },
    {
        "role": "system",
        "name": "example_assistant",
        "content": "Let's talk later when we're less busy about how to do better.",
    },
    {
        "role": "user",
        "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
    },
]

for model in [
    "gpt-3.5-turbo",
    "gpt-4-0613",
    "gpt-4",
    "gpt-4o",
    "gpt-4o-mini"
    ]:
    print(model)
    # example token count from the function defined above
    print(f"{num_tokens_from_messages(example_messages, model)} prompt tokens counted by num_tokens_from_messages().")
    # example token count from the OpenAI API
    response = client.chat.completions.create(model=model,
    messages=example_messages,
    temperature=0,
    max_tokens=1)
    if response.usage is not None:
        print(f'{response.usage.prompt_tokens} prompt tokens counted by the OpenAI API.')
    print()
