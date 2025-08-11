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
Preprocess the GSM8k dataset to parquet format
"""

import os
import json
import time
import random
import pandas as pd
import datasets


import argparse
MAX_LEN = 0

def add_prompt(table_desc, query):
        prompt_head = """
你是一位精通 Python 的数据分析师。你的任务是编写可执行的 Python 代码来解析表格，然后回答问题。

要求：
1. 根据问题，写出你的分析思路和方法，再根据这种方法写 Python 代码。
2. 请严格按照给你的文件路径和表格描述来生成代码。
3. 只生成一个代码块，并且严格以 ```python 开始，并以 ``` 结束。
4. 你的分析必须完全基于表格数据。如果用户的问题与数据分析无关，请礼貌拒绝。
5. 你需要生成可执行代码。如果有结果需要展示，请将结果存入answer函数中， 并用print展示。
6. 确保使用python库中pd.read_csv函数，读取给你的表格文件路径来进行数据处理，如需要读取多个表格，注意使用文件路径进行变量名区分。
7. 生成代码的过程中，请不要将数据转成DataFrame格式，请务必用pd.read_csv函数进行表格内容读取。

以下是提供的表格信息：
"""

        prompt_tail = """
确保最终答案是 Python 代码的最后一行，并且只能以 print(f'{answer}') 的形式出现，不能有其他形式。


让我们逐步思考，然后生成 Python 代码来分析表格并展示问题的最终答案。
输入问题：
"""

        query = prompt_head + "\n" + table_desc + "\n" + prompt_tail + query
        return query


# def add_prompt(table_desc, query):
#     prompt_head = """
# 你是一位精通 Python 的数据分析师，你的任务是通过给你的表格信息和问题编写可执行的 Python 代码来解析表格并回答问题。
#
# 要求：
# 1. 根据问题，写出你的分析思路和方法，再根据这种方法写 Python 代码。
# 2. 请严格按照给你的文件路径和表格描述来生成代码，只生成一个代码块，并且严格以 ```python 开始，并以 ``` 结束。
# 3. 注意理解表格结构，尤其是注意表头可能比较复杂需要小心处理。
# 4. 你的分析必须完全基于表格数据。如果用户的问题与数据分析无关，请礼貌拒绝。
# 5. 你需要生成可执行代码，将最终answer用print函数打印出来。
# 6. 确保使用python库中pd.read_csv函数来读取给你的表格文件路径，如需要读取多个表格，注意使用文件路径进行变量名区分。
#
# 以下是提供的表格信息：
# """
#
#     prompt_tail = """
#
# 让我们think step by step，然后生成 Python 代码来分析表格并展示问题的最终答案。
# 输入问题：
# """
#
#     query = prompt_head + "\n" + table_desc + "\n" + prompt_tail + query
#     return query


def change_data_distribution(data):
    new_data = []
    for line in data:
        if line['table_difficulty'] == 'medium':
            new_data += [line]*2
        elif line['table_difficulty'] == 'easy':
            new_data += [line]*3
        else:
            new_data.append(line)
    return new_data


def read_json_data(file, source, file_tag):
    if source=='normal_reasoning':
        data = []
        with open(file, 'r', encoding='utf-8') as fl:
            for line in fl:
                line = json.loads(line)
                line['problem'] = line['model_input'][0]['content']
                line['solution'] = line['reference_answer']
                line['category'] = source
                data.append(line)

    elif source=='table':
        all_data = []
        tables = None
        tags = None
        with open(file, 'r', encoding='utf-8') as fl:
            tables = json.load(fl)
        with open(file_tag, 'r', encoding='utf-8') as ft:
            tags = json.load(ft)
        for item, tag in zip(tables, tags):
            line = {}
            line['problem'] = item['instruction'] + item['input']
            line['solution'] = item['output']
            line['category'] = source
            line['tag'] = tag
            # print(item['output'], tag)
            all_data.append(line)
    else:
        data = []
        with open(file, 'r', encoding='utf-8') as fl:
            for line in fl:
                line = json.loads(line)
                line['category'] = source
                data.append(line)

    return all_data


if __name__ == '__main__':
    random.seed(2023)
    save_dir = "/fs-computility/llm_code_collab/liujiaheng/shudaixin/verl/data/mmt/grpo"
    local_test_dirs = ['/fs-computility/llm_code_collab/liujiaheng/shudaixin/LLaMA-Factory/data/mmt_test_full_no_think.json']
    local_train_dirs = ['/fs-computility/llm_code_collab/liujiaheng/shudaixin/LLaMA-Factory/data/mmt_train_full_think.json']
    
    local_test_tags = ['/fs-computility/llm_code_collab/liujiaheng/shudaixin/LLaMA-Factory/data/mmt_test_full_tags.json']
    local_train_tags = ['/fs-computility/llm_code_collab/liujiaheng/shudaixin/LLaMA-Factory/data/mmt_train_full_think_tags.json']

    train_data_sources = ['table']
    test_data_sources = ['table']

    all_train_dataset = []
    for f, source, f_tags in zip(local_train_dirs, train_data_sources, local_train_tags):
        train_dataset = read_json_data(f, source, f_tags)
        all_train_dataset.extend(train_dataset)
    all_test_dataset = []
    for f, source, f_tags in zip(local_test_dirs, test_data_sources, local_test_tags):
        test_dataset = read_json_data(f, source, f_tags)
        all_test_dataset.extend(test_dataset)

    random.shuffle(all_train_dataset)
    random.shuffle(all_test_dataset)

    # if there is no test data, get test data from train data
    if len(all_test_dataset) == 0:
        all_test_dataset = all_train_dataset[:100]
        all_train_dataset = all_train_dataset[100:]

    # add a row to each data item that represents a unique id
    def make_map_fn(split, example, idx):
        question = example['problem']
        solution = example['solution']
        tag = example['tag']
        tag['atype'] = str(tag['atype'])
        
        global MAX_LEN
        if MAX_LEN < len(question):
            MAX_LEN = len(question)

        data = {
            "data_source": example['category'],     # 'table'
            "prompt": [{                            # query
                "role": "user",
                "content": question
            }],
            "ability": "table",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution            # gold_truth
            },
            "extra_info": {
                'split': split,                     # train/test
                'index': idx,
                'tag': tag
            }
        }

        # print(data)

        return data

    train_dataset = [make_map_fn('train', line, idx) for idx, line in enumerate(all_train_dataset)]
    test_dataset = [make_map_fn('test', line, idx) for idx, line in enumerate(all_test_dataset)]

    print(len(train_dataset), len(test_dataset))

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"Directory '{save_dir}' created successfully")
    else:
        print(f"Directory '{save_dir}' already exists")

    train_dataset = pd.DataFrame(train_dataset)
    test_dataset = pd.DataFrame(test_dataset)

    print(train_dataset)
    print(MAX_LEN)

    train_dataset.to_parquet(os.path.join(save_dir, 'train_think.parquet'), index=0)
    test_dataset.to_parquet(os.path.join(save_dir, 'test.parquet'), index=0)

