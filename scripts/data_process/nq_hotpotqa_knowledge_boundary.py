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
"""
Preprocess the nq dataset to parquet format
"""
from datasets import concatenate_datasets
import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = \
f"""You are an expert assistant capable of solving knowledge-intensive tasks efficiently. You will be given a question to answer as accurately as possible.

You can use your own knowledge or call external search engines to gather additional information, but searching should only occur when necessary. Specifically, you should search only when encountering a clear knowledge gap or uncertainty that prevents you from confidently answering the question.

To arrive at the answer, you will proceed step-by-step in a structured cycle of '<think>thinking content</think>', '<search>search query</search>' (optional), and '<information>returned external information</information>' (optional) sequences. At end, you should use <answer> final answer </answer> to provide answer. You can only generate content within these special tags.
Remember that <search>xxx</search> and <information>xxx</information> are optional. You can skip them if you have enough knowledge to answer the question. And skip is them is encouraged and preferable.
Thinking Phase (<think>): For question, it may be decomposed into sub-questions for you to think about. Some sub-questions may be answered by searching, while others may not. You can also use the <think> tag to express your uncertainty about the sub-question. Recall your own knowledge, analyze current information, and decide whether further search is needed. If enough knowledge is available, skip searching. 
Searching Phase (<search>): Formulate a search query only if required to fill a knowledge gap or verify uncertainty. Skip if unnecessary.
Information Phase (<information>): Use search results as context for further steps. If no search was performed, proceed without this phase.
Answering Phase (<answer>): Provide a concise and accurate answer within <answer> tags once you have enough knowledge. The answer should be short and precise, such as <answeer> Beijing </answer>.

Here are some examples:
---
Example 1: search is needed, search more than once
Question: xxx

<think> xxx </think>
<search> xxx </search>
<information> xxx </information>
<think> xxx </think>
... (search more than once)
<think> xxx </think>
<answer> xx </answer>

---
Example 2: search is needed, only search once
Question: xxx?

<think> xxx </think>
<search> xxx </search>
<information> xxx </information>
<think> xxx </think>
<answer> xx </answer>

---
Example 3: search is not needed
Question: xxx?

<think> xxx </think>
<answer> xxx </answer>
---

Less search is preferable. Each search should be foucsed on one sub-question.
The answer within <answer> tags should be short and precise, such as <answer> yes </answer>.
Now it is your turn to answer the question.
Question: {question}\n
"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/nq_hotpotqa_knowledge_boundary')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'nq'
    
    # nq_dataset = datasets.load_dataset("hzy/Qwen2.5-3B-Instruct-nq-knowledge-probe-balanced")['train']
    nq_easy_dataset = datasets.load_dataset("hzy/ikea_nq_easy")
    nq_hard_dataset = datasets.load_dataset("hzy/ikea_nq_hard")
    nq_train = concatenate_datasets([nq_easy_dataset['train'], nq_hard_dataset['train']])
    nq_test = concatenate_datasets([nq_easy_dataset['test'], nq_hard_dataset['test']])

    hotpotqa_easy_dataset = datasets.load_dataset("hzy/ikea_hotpotqa_easy")
    hotpotqa_hard_dataset = datasets.load_dataset("hzy/ikea_hotpotqa_hard")
    hotpotqa_train = concatenate_datasets([hotpotqa_easy_dataset['train'], hotpotqa_hard_dataset['train']])
    hotpotqa_test = concatenate_datasets([hotpotqa_easy_dataset['test'], hotpotqa_hard_dataset['test']])
    # nq_train_know = nq_dataset['train'].filter(lambda example: example['pass_rate'] > 0).shuffle(42).select(range(4000))
    # nq_train_unknow = nq_dataset['train'].filter(lambda example: example['pass_rate'] == 0).shuffle(42).select(range(4000))
    # nq_train = concatenate_datasets([nq_train_unknow, nq_train_know]).shuffle(42)

    # nq_test_know = nq_dataset['test'].filter(lambda example: example['pass_rate'] > 0).shuffle(42).select(range(512))
    # nq_test_unknow = nq_dataset['test'].filter(lambda example: example['pass_rate'] == 0).shuffle(42).select(range(512))
    # nq_test = concatenate_datasets([nq_test_unknow, nq_test_know]).shuffle(42)

    # hotpotqa_dataset = datasets.load_dataset("hzy/hotpotqa_know_probe_16")
    # hotpotqa_train_know = hotpotqa_dataset['train'].filter(lambda example: example['pass_rate'] > 0).shuffle(42).select(range(4000))
    # hotpotqa_train_unknow = hotpotqa_dataset['train'].filter(lambda example: example['pass_rate'] == 0).shuffle(42).select(range(4000))
    # hotpotqa_train = concatenate_datasets([hotpotqa_train_unknow, hotpotqa_train_know]).shuffle(42)

    # hotpotqa_test_know = hotpotqa_dataset['test'].filter(lambda example: example['pass_rate'] > 0).shuffle(42).select(range(512))
    # hotpotqa_test_unknow = hotpotqa_dataset['test'].filter(lambda example: example['pass_rate'] == 0).shuffle(42).select(range(512))
    # hotpotqa_test = concatenate_datasets([hotpotqa_test_unknow, hotpotqa_test_know]).shuffle(42)

    # dataset = datasets.load_dataset("hzy/Qwen2.5-3B-Instruct-nq-knowledge-probe-balanced")
    # hotpot_dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', 'hotpotqa')
    # hotpot_train = hotpot_dataset['train'].shuffle(42).select(range(8000))
    # hotpot_test = hotpot_dataset['dev'].shuffle(42).select(range(1024))
    
    train_dataset = concatenate_datasets([nq_train, hotpotqa_train]).shuffle(42)
    test_dataset = concatenate_datasets([nq_test, hotpotqa_test]).shuffle(42)

    # train_dataset = dataset['train']
    # train_dataset = train_dataset.shuffle(42).select(range(8000))
    # # train_dataset = train_dataset.filter(lambda e: e['know_or_unknow'] == "know")
    # # train_dataset = train_dataset.shuffle(42).select(range(8000))
    # test_dataset = dataset['dev'].shuffle(42).select(range(1024))
    print(f"train data length: {len(train_dataset)}")
    print(f"test data length: {len(test_dataset)}")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = {
                "target": example['golden_answers'],
            }

            if split == 'train':
                data = {
                    "data_source": "nq_kb",
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'know_or_unknow': "know"
                    }
                }
            elif split == 'test':
                data = {
                    "data_source": "nq_kb",
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        'know_or_unknow': "null"
                    }
                }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
