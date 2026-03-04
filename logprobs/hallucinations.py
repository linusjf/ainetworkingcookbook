#!/usr/bin/env python
"""
Hallucinations.

######################################################################
# @author      : Linus Fernandes (linusfernandes at gmail dot com)
# @file        : hallucinations
# @created     : Wednesday Mar 04, 2026 10:30:25 IST
# @description : Example of how to reduce hallucinations
# -*- coding: utf-8 -*-'
######################################################################
"""

from openai import OpenAI
import json

client = OpenAI(api_key="YOUR_API_KEY")

# Step 1: Evaluation prompt
def check_content_sufficiency(content, user_query):
    prompt = f"""
    Use ONLY this content:
    {content}

    First decide whether the content can answer this inquiry.
    Return JSON exactly like:
    {{
      "content_contains_answer": boolean,
      "justification": string
    }}

    Inquiry: {user_query}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    # parse the JSON
    text = resp.choices[0].message["content"]
    return json.loads(text)

# Step 2: Generate the final answer only if evaluation says true
def generate_answer(content, user_query):
    eval_res = check_content_sufficiency(content, user_query)

    if not eval_res["content_contains_answer"]:
        return {"answer":"", "reason":eval_res["justification"]}

    # safe to run actual answer
    answer_prompt = f"""
    Use ONLY this content:
    {content}

    Provide final answer to:
    {user_query}
    """
    resp2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": answer_prompt}],
        max_tokens=300
    )

    return {"answer": resp2.choices[0].message["content"],
            "justification": eval_res["justification"]}

def confident_answer(content, user_query, threshold=0.9):
    ans_data = generate_answer(content, user_query)

    # if there's no answer, just return early
    if not ans_data["answer"]:
        return ans_data

    # run low-confidence detection using logprobs
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":ans_data["answer"]}],
        max_tokens=0,
        logprobs=True  # ask API to return logprobs
    )

    # compute simple probability metric
    token_logprob = resp.choices[0].logprobs["token_logprobs"]
    # convert average to probability
    avg_prob = sum([pow(10, lp) for lp in token_logprob]) / len(token_logprob)

    if avg_prob < threshold:
        # low confidence: mark as potentially hallucinated
        return {"answer": "",
                "reason": "Low confidence – requires user review or further grounding."}

    return ans_data

def classify_intent(user_query):
    prompt = f"""
    Classify the intent of this query in JSON:
    {{
      "intent": "classification label"
    }}

    Query: {user_query}
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )

    return json.loads(resp.choices[0].message["content"])["intent"]

def pipeline(content, user_query):
    intent = classify_intent(user_query)
    # choose different evaluation prompt based on intent
    if intent == "technical":
        # maybe inject more structured instructions
        pass

    return confident_answer(content, user_query)
