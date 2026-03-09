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
import numpy as np
import argparse

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
        print("Top log probabilities:")
        for i, logprob in enumerate(top_two_logprobs, start=1):
            linear_prob = np.round(np.exp(logprob.logprob) * 100, 2)
            print(f"  Output token {i}: '{logprob.token}', "
                  f"logprob: {logprob.logprob:.4f}, "
                  f"linear probability: {linear_prob}%")
        print()


def retrieval_confidence_scoring():
    # Article retrieved
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

    PROMPT = """You retrieved this article: {article}. The question is: {question}.
Before even answering the question, consider whether you have sufficient information in the article to answer the question fully.
Your output should JUST be the boolean true or false, of if you have sufficient information in the article to answer the question.
Respond with just one word, the boolean true or false. You must output the word 'True', or the word 'False', nothing else.
"""

    print("\n" + "="*80)
    print("Questions clearly answered in article")
    print("="*80)

    for question in easy_questions:
        API_RESPONSE = get_completion(
            [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        article=ada_lovelace_article, question=question
                    ),
                }
            ],
            model="gpt-4o",
            logprobs=True,
        )
        print(f"\nQuestion: {question}")
        for logprob in API_RESPONSE.choices[0].logprobs.content:
            linear_prob = np.round(np.exp(logprob.logprob) * 100, 2)
            print(f"  has_sufficient_context_for_answer: '{logprob.token}', "
                  f"logprob: {logprob.logprob:.4f}, "
                  f"linear probability: {linear_prob}%")

    print("\n" + "="*80)
    print("Questions only partially covered in the article")
    print("="*80)

    for question in medium_questions:
        API_RESPONSE = get_completion(
            [
                {
                    "role": "user",
                    "content": PROMPT.format(
                        article=ada_lovelace_article, question=question
                    ),
                }
            ],
            model="gpt-4o",
            logprobs=True,
            top_logprobs=3,
        )
        print(f"\nQuestion: {question}")
        for logprob in API_RESPONSE.choices[0].logprobs.content:
            linear_prob = np.round(np.exp(logprob.logprob) * 100, 2)
            print(f"  has_sufficient_context_for_answer: '{logprob.token}', "
                  f"logprob: {logprob.logprob:.4f}, "
                  f"linear probability: {linear_prob}%")

        # Also show top logprobs if available
        if hasattr(API_RESPONSE.choices[0].logprobs.content[0], 'top_logprobs'):
            print("  Top log probabilities:")
            for i, top_logprob in enumerate(API_RESPONSE.choices[0].logprobs.content[0].top_logprobs, start=1):
                top_linear_prob = np.round(np.exp(top_logprob.logprob) * 100, 2)
                print(f"    Alternative {i}: '{top_logprob.token}', "
                      f"logprob: {top_logprob.logprob:.4f}, "
                      f"linear probability: {top_linear_prob}%")


def autocomplete():
    sentence_list = [
    "My",
    "My least",
    "My least favorite",
    "My least favorite TV",
    "My least favorite TV show",
    "My least favorite TV show is",
    "My least favorite TV show is Breaking Bad",
    ]

    high_prob_completions = {}
    low_prob_completions = {}

    print("\n" + "="*80)
    print("Autocomplete Predictions")
    print("="*80)

    for sentence in sentence_list:
        PROMPT = """Complete this sentence. You are acting as auto-complete. Simply complete the sentence to the best of your ability, make sure it is just ONE sentence: {sentence}"""
        API_RESPONSE = get_completion(
            [{"role": "user", "content": PROMPT.format(sentence=sentence)}],
            model="gpt-4o",
            logprobs=True,
            top_logprobs=3,
        )
        print(f"\nSentence: {sentence}")
        first_token = True
        for i, token in enumerate(API_RESPONSE.choices[0].logprobs.content[0].top_logprobs, start=1):
            linear_prob = np.round(np.exp(token.logprob) * 100, 2)
            print(f"  Predicted next token {i}: '{token.token}', "
                  f"logprob: {token.logprob:.4f}, "
                  f"linear probability: {linear_prob}%")
            if first_token:
                linear_prob = np.exp(token.logprob)
                if linear_prob > 0.95:
                    high_prob_completions[sentence] = token.token
                if linear_prob < 0.60:
                    low_prob_completions[sentence] = token.token
            first_token = False

    # Print summary of high and low probability completions
    print("\n" + "="*80)
    print("Summary of High Probability Completions (>95%)")
    print("="*80)
    for sentence, token in high_prob_completions.items():
        print(f"  '{sentence}' -> '{token}'")

    print("\n" + "="*80)
    print("Summary of Low Probability Completions (<60%)")
    print("="*80)
    for sentence, token in low_prob_completions.items():
        print(f"  '{sentence}' -> '{token}'")

def highlight_bytes():
    PROMPT = """What's the longest word in the English language?"""

    API_RESPONSE = get_completion(
        [{"role": "user", "content": PROMPT}], model="gpt-4o", logprobs=True, top_logprobs=5
    )


    def highlight_text(api_response):
        # ANSI color codes for terminal output
        colors = [
            "\033[95m",  # Magenta
            "\033[92m",  # Green
            "\033[93m",  # Yellow
            "\033[91m",  # Red
            "\033[94m",  # Blue
        ]
        reset_color = "\033[0m"

        tokens = api_response.choices[0].logprobs.content

        color_idx = 0  # Initialize color index
        print("\n" + "="*80)
        print("Token Highlighting (Colored by Token)")
        print("="*80)
        print("\nResponse text with colored tokens:")
        print()

        for t in tokens:
            token_str = bytes(t.bytes).decode("utf-8")  # Decode bytes to string

            # Print colored token to console
            print(f"{colors[color_idx]}{token_str}{reset_color}", end="")

            # Move to the next color
            color_idx = (color_idx + 1) % len(colors)

        print("\n\n" + "="*80)
        print("Token Details")
        print("="*80)

        print(f"\nTotal number of tokens: {len(tokens)}")

    highlight_text(API_RESPONSE)

def main():
    parser = argparse.ArgumentParser(description="Run logprobs demonstration functions")
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Run news article classification demonstration"
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Run retrieval confidence scoring demonstration"
    )
    parser.add_argument(
        "--autocomplete",
        action="store_true",
        help="Run autocomplete demonstration"
    )
    parser.add_argument(
        "--highlight",
        action="store_true",
        help="Run token highlighting demonstration"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all demonstrations"
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.classify, args.retrieval, args.autocomplete, args.highlight, args.all]):
        parser.print_help()
        return
    
    # Run selected functions
    if args.all or args.classify:
        print("\n" + "="*80)
        print("Running: News Article Classification")
        print("="*80)
        classify_news_articles()
    
    if args.all or args.retrieval:
        print("\n" + "="*80)
        print("Running: Retrieval Confidence Scoring")
        print("="*80)
        retrieval_confidence_scoring()
    
    if args.all or args.autocomplete:
        print("\n" + "="*80)
        print("Running: Autocomplete Demonstration")
        print("="*80)
        autocomplete()
    
    if args.all or args.highlight:
        print("\n" + "="*80)
        print("Running: Token Highlighting")
        print("="*80)
        highlight_bytes()

if __name__ == "__main__":
    main()
