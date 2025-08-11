import os
import re
import json
import ast
import argparse
import requests
from rouge import Rouge


def get_between_strings_regex(text, str1, str2):
    # pattern = f"{re.escape(str1)}(.*?){re.escape(str2)}"
    # match = re.search(pattern, text)

    pattern = re.compile(f"{re.escape(str1)}(.*?){re.escape(str2)}", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None

def get_between_multi(text, start_str, end_str):
    pattern = re.compile(f"{re.escape(start_str)}(.*?){re.escape(end_str)}")
    return pattern.findall(text)

def set_accuracy(y_true, y_pred):
    set_true = set(y_true)
    set_pred = set(y_pred)
    intersection = set_true & set_pred
    
    # 准确率 = 共有元素种类数 / 真实元素种类数
    return len(intersection) / len(set_true) if set_true else 0.0

def jaccard_similarity(list1, list2):
    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)
    
    # 计算交集和并集的大小
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # 计算Jaccard相似度
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0

    return jaccard_index

def get_digit_score(answer, annotation):
    answer_list = []
    annotation_list = []

    answer = get_between_strings_regex(answer, '[', ']')
    for s in answer.split(','):
        s = s.strip()
        s = float(s[:-1])/100 if s.endswith("%") else float(s)
        answer_list.append(s)

    annotation = get_between_strings_regex(annotation, '[', ']')
    for s in annotation.split(','):
        s = s.strip()
        s = float(s[:-1])/100 if s.endswith("%") else float(s)
        annotation_list.append(s)

    return jaccard_similarity(annotation_list, answer_list)



def get_locate_scores(annotation, answer):

    annotation = annotation.replace("{\'", "{\"").replace("\', \'","\", \"").replace("\'}", 
                                    "\"}").replace("\': \'", "\": \"").replace("{{", "{").replace("}}", 
                                    "}").replace("], [", ", ")
    answer = answer.replace("{\'", "{\"").replace("\', \'","\", \"").replace("\'}", 
                                    "\"}").replace("\': \'", "\": \"").replace("{{", "{").replace("}}", 
                                    "}").replace("], [", ", ")

    # annotation = annotation.replace("\'", "\"")
    # answer = answer.replace("\'", "\"")

    annotation_list = json.loads(annotation)
    annotation_list = [item["locate"] for item in annotation_list]

    answer_list = json.loads(answer)
    answer_list = [item["locate"] for item in answer_list]
    return jaccard_similarity(annotation_list, answer_list)

def compute_score(solution_str, ground_truth):
    # print('solution', solution_str)
    # print('ground_truth', ground_truth)
    predict_answer = get_between_multi(solution_str, '<answer>', '</answer>')
    if '<answer>' in ground_truth:
        bench_answer = get_between_strings_regex(ground_truth, '<answer>', '</answer>')
    else:
        bench_answer = ground_truth
    score = 0

    for pre_answer in predict_answer:
        if bench_answer[0] == '[':
            if '{' in bench_answer:
                # 单元格定位
                try:
                    locate_score=get_locate_scores(pre_answer, bench_answer)
                    score=score if score > locate_score else locate_score
                except Exception as e:
                    print(e)
                    print('2——————', pre_answer,'——————', bench_answer)
            else:
                # 数值计算
                try:
                    digit_score = get_digit_score(pre_answer, bench_answer)
                    score=score if score > digit_score else digit_score
                    # print('1')
                except Exception as e:
                    print(e)
                    print('1——————', pre_answer,'——————', bench_answer)

            
        elif bench_answer.upper() == 'T' or bench_answer.upper() == 'F':
            # 对错
            true_false_scores = 1 if pre_answer.lower() == bench_answer.lower() else 0
            score=score if score > true_false_scores else true_false_scores
        else:
            try:
                rouge = Rouge()
                # print(pre_answer)
                # print(bench_answer)
                scores = rouge.get_scores(pre_answer, bench_answer, avg=True)
                # print(i,scores['rouge-1']['f'])
                score=score if score > scores['rouge-1']['f'] else scores['rouge-1']['f']
            except Exception as e:
                print(e)
                print('4——————', pre_answer,'——————', bench_answer)
    return score