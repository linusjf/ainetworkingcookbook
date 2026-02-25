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

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

print(encoding)

print(f"Encoding '{token_string}'")
encoded = encoding.encode(token_string)
print(encoded)
decoded_string = encoding.decode(encoded)
print(f"Decoded {encoded} to '{decoded_string}'")

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

encoding_name = "o200k_base"
print(f"Getting number of tokens for string '{token_string}' using encoding '{encoding_name}'")
token_count = num_tokens_from_string(token_string, encoding_name)
print(f"Token count is {token_count}")
