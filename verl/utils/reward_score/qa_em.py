# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
from collections import Counter

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_single(prediction: str, ground_truth: str, tokenfun=lambda x: x.split()):
    prediction_tokens = tokenfun(normalize_answer(prediction))
    ground_truth_tokens = tokenfun(normalize_answer(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def f1_score(prediction: str, golden_answers: list[str], tokenfun=lambda x: x.split()):
    f1_, precision_, recall_ = [max(values) for values in zip(*[f1_single(prediction, gt, tokenfun) for gt in golden_answers])]
    return f1_


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def validate_format(s):
    pattern = r"<think>.*?</think>(?:.*?<search>.*?</search>.*?<think>.*?</think>)*.*?<answer>.*?</answer>"
    return re.search(pattern, s, re.DOTALL) is not None

delimiter_str = "Now it is your turn to answer the question."

def compute_score_em(solution_str, ground_truth, know_or_unknow, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    solution_str = solution_str.split(delimiter_str)[-1]
    flag = True
    if "My previous action is invalid" in solution_str:
        flag = False

    # if not validate_format(solution_str):
    #     flag = False

    if "<think>" not in solution_str or "</think>" not in solution_str:
        flag = False
        
    if "(think" in solution_str or "(search" in solution_str or "(answer" in solution_str:
        flag = False
        
    if "think)" in solution_str or "search)" in solution_str or "answer)" in solution_str:
        flag = False
    
    if "[think" in solution_str or "[search" in solution_str or "[answer" in solution_str:
        flag = False
    
    if "think]" in solution_str or "search]" in solution_str or "answer]" in solution_str:
        flag = False
    
    search_times = solution_str.count("<search>")
    kb_score = 0
    if search_times == 0:
        kb_score = 0.6
    elif search_times == 1:
        kb_score = 0.4
    elif search_times == 2:
        kb_score = 0.2
    elif search_times >= 3:
        kb_score = 0
        
    train_or_test = None
    if know_or_unknow in ['know', 'unknow']:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        
        
    answer_score = 0
    if answer is None:
        answer_score = 0
    else:
        if em_check(answer, ground_truth['target']):
            answer_score = 1
        else:
            answer_score = 0
        # answer_score = f1_score(answer, ground_truth['target'])
    
    if train_or_test == 'train':
        if flag:
            if answer_score > 0:
                # return answer_score
                return answer_score + kb_score
            else:
                if "<search>" in solution_str:
                    return 0.05
                return 0
            # return answer_score + kb_score
        else:
            return -1
    else:
        if answer_score == 1:
            return 1
        else:
            return 0
        # return answer_score



def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
