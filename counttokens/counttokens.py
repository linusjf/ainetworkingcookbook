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
