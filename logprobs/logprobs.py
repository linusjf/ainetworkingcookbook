#!/usr/bin/env python
"""
Logprobs.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : logprobs
# @created     : Friday Feb 27, 2026 15:38:59 IST
# @description :
# -*- coding: utf-8 -*-'
######################################################################
"""
from openai import OpenAI
from typing import Any
from math import exp
import numpy as np
from IPython.display import display, HTML
import os

client = OpenAI()

def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 500,
    temperature: float = 0,
    seed: int | None = 123,
    tools: list[dict[str, Any]] | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
):
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }

    if tools is not None:
        params["tools"] = tools

    return client.chat.completions.create(**params)


def classify_news_articles():
    CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
Return only the name of the category, and nothing else.
MAKE SURE your output is one of the four categories stated.
Article headline: {headline}"""

    headlines: list[str] = [
        "Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.",
        "Local Mayor Launches Initiative to Enhance Urban Public Transport.",
        "Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut",
    ]

    for headline in headlines:
        print(f"\nHeadline: {headline}")

        response = get_completion(
            messages=[
                {
                    "role": "user",
                    "content": CLASSIFICATION_PROMPT.format(headline=headline),
                }
            ],
            model="gpt-4o",
        )

        print(f"Category: {response.choices[0].message.content}\n")

    for headline in headlines:
        print(f"\nHeadline: {headline}")
        API_RESPONSE = get_completion(
            [{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
            model="gpt-4o",
            logprobs=True,
            top_logprobs=2,
        )
        top_two_logprobs = API_RESPONSE.choices[0].logprobs.content[0].top_logprobs
        html_content = ""
        for i, logprob in enumerate(top_two_logprobs, start=1):
            html_content += (
                f"<span style='color: cyan'>Output token {i}:</span> {logprob.token}, "
                f"<span style='color: darkorange'>logprobs:</span> {logprob.logprob}, "
                f"<span style='color: magenta'>linear probability:</span> {np.round(np.exp(logprob.logprob)*100,2)}%<br>"
            )
        display(HTML(html_content))
        print("\n")
