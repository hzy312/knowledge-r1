"""Reward functions for GRPO training."""

import math
import re
import regex
from collections import Counter
import string
from dataclasses import dataclass, field
from typing import List
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    # else:
        # print("  Tag sequence validation passed")
        # pass

    return validation_passed

def validate_search_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    # print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<search>', 1),
        'answer_end': ('</search>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        # print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            # print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    # else:
        # print("  Tag sequence validation passed")
        # pass

    return validation_passed

# def accuracy_reward(completions, solution, **kwargs):
#     """Reward function that checks if the completion is the same as the ground truth."""
#     contents = [completion[0]["content"] for completion in completions]
#     rewards = []
#     for content, sol in zip(contents, solution):
#         gold_parsed = parse(
#             sol,
#             extraction_mode="first_match",
#             extraction_config=[LatexExtractionConfig()],
#         )
#         if len(gold_parsed) != 0:
#             # We require the answer to be provided in correct latex (no malformed operators)
#             answer_parsed = parse(
#                 content,
#                 extraction_config=[
#                     LatexExtractionConfig(
#                         normalization_config=NormalizationConfig(
#                             nits=False,
#                             malformed_operators=False,
#                             basic_latex=True,
#                             equations=True,
#                             boxed="all",
#                             units=True,
#                         ),
#                         # Ensures that boxed is tried first
#                         boxed_match_priority=0,
#                         try_extract_without_anchor=False,
#                     )
#                 ],
#                 extraction_mode="first_match",
#             )
#             # Reward 1 if the content is the same as the ground truth, 0 otherwise
#             reward = float(verify(answer_parsed, gold_parsed))
#         else:
#             # If the gold solution is not parseable, we reward 1 to skip this example
#             reward = 1.0
#             print("Failed to parse gold solution: ", sol)
#         rewards.append(reward)

#     return rewards

def normalize(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_single(prediction: str, ground_truth: str, tokenfun=lambda x: x.split()):
    prediction_tokens = tokenfun(normalize(prediction))
    ground_truth_tokens = tokenfun(normalize(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def f1_score(predictions: list[str], references: list[list[str]], tokenfun=lambda x: x.split()):
    f1, precision, recall = list(), list(), list()
    for ground_truths, prediction in zip(references, predictions):
        f1_, precision_, recall_ = [max(values) for values in zip(*[f1_single(prediction, gt, tokenfun) for gt in ground_truths])]
        f1.append(f1_)
        precision.append(precision_)
        recall.append(recall_)
    return {"f1": f1, "precision": precision, "recall": recall}

def em_single(prediction: str, ground_truth: str):
    return float(normalize(prediction) == normalize(ground_truth))


def exact_match_score(predictions: list[str], references: list[list[str]]):
    match_samples = [max([em_single(prediction, gt) for gt in ground_truths]) for ground_truths, prediction in zip(references, predictions)] 
    return match_samples


def accuracy_reward(completions, golden_answers, msg_lists, **kwargs):
    """
    Reward function that checks if the completion is the same as the ground truth.
    This is a knowledge-intensive question answering task.
    """
    contents = [msg[-1]["content"] for msg in msg_lists]
    # for each content, the format is like this: <think> reasoning process here </think><answer> answer here </answer>
    # extract the answer part
    contents = [re.search(r'<answer>(.*?)</answer>', content, re.DOTALL).group(1).strip() if re.search(r'<answer>(.*?)</answer>', content, re.DOTALL) else "None" for content in contents]
    rewards = []
    em_scores = exact_match_score(contents, golden_answers)
    return em_scores
    # f1_scores = f1_score(contents, golden_answers)["f1"]
    # rewards = [2 if em_score == 1.0 else f1_score for em_score, f1_score in zip(em_scores, f1_scores)]
    # rewards = [reward if context != "None" else 0 for reward, context in zip(rewards, contents) ]
    # rewards = f1_scores
    # return rewards



def format_reward(completions, msg_lists, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    # return [1.0 if match else 0.0 for match in matches]
    reward = [True for msg in msg_lists]
    for i, msg_list in enumerate(msg_lists):
        for msg in msg_list[2:-1:2]:
            assert msg['role'] == "assistant"
            if not validate_search_structure(msg["content"]):
                reward[i] = False
                break            
        if not validate_response_structure(msg_list[-1]["content"]):
            assert msg_list[-1]['role'] == "assistant"
            reward[i] = False
    return [1.0 if r else -1.0 for r in reward]
        
        
    # contents = [completion[0]["content"] for completion in completions]
    # rewards = [1.0 if validate_response_structure(content) else -1.0 for content in contents]
    # return rewards


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward
