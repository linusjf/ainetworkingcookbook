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
import math

client = OpenAI()

# Step 1: Evaluation prompt
import json

def check_content_sufficiency(content: str, user_query: str) -> dict:

    prompt = f"""
Return ONLY valid JSON.

Schema:
{{"content_contains_answer": boolean, "justification": string}}

Content:
{content}

Question:
{user_query}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    msg = resp.choices[0].message
    text = msg.content

    if not text:
        raise RuntimeError("Model returned empty output")

    print(f"Content sufficiency: {text}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Model returned invalid JSON: {text}")

# Step 2: Generate the final answer only if evaluation says true

def generate_answer_with_evidence(content: str, question: str):

    prompt = f"""
Use ONLY the provided content.

Return JSON.

Schema:
{{
  "answer": string,
  "evidence": [string]
}}

Evidence must be exact phrases copied from the content.

Content:
{content}

Question:
{question}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=1,
        max_tokens=200
    )

    msg = resp.choices[0].message
    data = json.loads(msg.content) if msg.content is not None else None
    print(f"Generated data: {data}")
    return data, resp.choices[0]

def evidence_logprob(choice, evidence_spans):

    print(f"Evidence spans: {evidence_spans}")
    tokens = choice.logprobs.content

    evidence_tokens = []

    for tok in tokens:
        token = tok.token.strip()

        for span in evidence_spans:
            if token and token in span:
                evidence_tokens.append(tok.logprob)
                break


    print(f"Evidence tokens log probabilities: {evidence_tokens}")
    if not evidence_tokens:
        return None

    return sum(evidence_tokens) / len(evidence_tokens)

def confident_answer(content, question):


    data, choice = generate_answer_with_evidence(content, question)
    if data is not None:
        score = evidence_logprob(choice, data["evidence"])

        if score is None or score < -0.8:
            return {
                "answer": "",
                "reason": "Evidence tokens low confidence"
            }

        return {
            "answer": data["answer"],
            "confidence": score
        }

    return None

def classify_intent(user_query: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": f"""
Return the result as JSON.

Query: {user_query}

Schema:
{{ "intent": "string" }}
"""
        }]
    )

    msg = resp.choices[0].message
    assert msg.content is not None

    return json.loads(msg.content)["intent"]

def pipeline(content, user_query):
    intent = classify_intent(user_query)
    # choose different evaluation prompt based on intent
    if intent == "technical":
        # maybe inject more structured instructions
        pass
    print(f"Intent: {intent}")
    return confident_answer(content, user_query)

def main():
    ada_lovelace_article = """Augusta Ada King, Countess of Lovelace (née Byron; 10 December 1815 – 27 November 1852) was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine. She was the first to recognise that the machine had applications beyond pure calculation.
    Ada Byron was the only legitimate child of poet Lord Byron and reformer Lady Byron. All Lovelace's half-siblings, Lord Byron's other children, were born out of wedlock to other women. Byron separated from his wife a month after Ada was born and left England forever. He died in Greece when Ada was eight. Her mother was anxious about her upbringing and promoted Ada's interest in mathematics and logic in an effort to prevent her from developing her father's perceived insanity. Despite this, Ada remained interested in him, naming her two sons Byron and Gordon. Upon her death, she was buried next to him at her request. Although often ill in her childhood, Ada pursued her studies assiduously. She married William King in 1835. King was made Earl of Lovelace in 1838, Ada thereby becoming Countess of Lovelace.
    Her educational and social exploits brought her into contact with scientists such as Andrew Crosse, Charles Babbage, Sir David Brewster, Charles Wheatstone, Michael Faraday, and the author Charles Dickens, contacts which she used to further her education. Ada described her approach as "poetical science" and herself as an "Analyst (& Metaphysician)".
    When she was eighteen, her mathematical talents led her to a long working relationship and friendship with fellow British mathematician Charles Babbage, who is known as "the father of computers". She was in particular interested in Babbage's work on the Analytical Engine. Lovelace first met him in June 1833, through their mutual friend, and her private tutor, Mary Somerville.
    Between 1842 and 1843, Ada translated an article by the military engineer Luigi Menabrea (later Prime Minister of Italy) about the Analytical Engine, supplementing it with an elaborate set of seven notes, simply called "Notes".
    Lovelace's notes are important in the early history of computers, especially since the seventh one contained what many consider to be the first computer program—that is, an algorithm designed to be carried out by a machine. Other historians reject this perspective and point out that Babbage's personal notes from the years 1836/1837 contain the first programs for the engine. She also developed a vision of the capability of computers to go beyond mere calculating or number-crunching, while many others, including Babbage himself, focused only on those capabilities. Her mindset of "poetical science" led her to ask questions about the Analytical Engine (as shown in her notes) examining how individuals and society relate to technology as a collaborative tool.
    """

    # Questions that can be easily answered given the article
    easy_questions = [
        "What nationality was Ada Lovelace?",
        "What was an important finding from Lovelace's seventh note?",
    ]

    # Questions that are not fully covered in the article
    medium_questions = [
        "Did Lovelace collaborate with Charles Dickens",
        "What concepts did Lovelace build with Charles Babbage",
    ]


    print("\n" + "="*80)
    print("Questions clearly answered in article")
    print("="*80)

    for question in easy_questions:
        print(f"Question: {question}")
        print(pipeline(ada_lovelace_article, question))
        print()

    print("\n" + "="*80)
    print("Questions only partially covered in the article")
    print("="*80)

    for question in medium_questions:
        print(f"Question: {question}")
        print(pipeline(ada_lovelace_article, question))
        print()

if __name__ == "__main__":
    main()
