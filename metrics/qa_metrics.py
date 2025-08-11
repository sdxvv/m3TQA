import os
import re
import json
import csv
import ast
import argparse
import requests
from rouge import Rouge


def get_between_strings_regex(text, str1, str2):
    text = text.lower()
    # pattern = f"{re.escape(str1)}(.*?){re.escape(str2)}"
    # match = re.search(pattern, text)

    pattern = re.compile(f"{re.escape(str1)}(.*?){re.escape(str2)}", re.DOTALL) # re.DOTALL 启用单行模式
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None

def get_between_multi(text, start_str, end_str):
    text = text.lower()
    pattern = re.compile(f"{re.escape(start_str)}(.*?){re.escape(end_str)}", re.DOTALL)
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

    if '[' in answer and ']' in answer:
        answer = answer.strip()
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


def replace_double_quota(text):
    # text = text.replace("\"", "'")
    text = text.replace("{'", "{\"").replace("', '","\", \"").replace("'}", "\"}").replace("': '", "\": \"")
    text = text.replace("{\\'", "{\"").replace("\\', \\'","\", \"").replace("\\'}", "\"}").replace("\\': \\'", "\": \"")
    return text

def get_locate_scores(annotation, answer):
    aaa = answer
    bbb = annotation
    annotation = replace_double_quota(annotation)
    annotation = annotation.replace("{{", "{").replace("}}", 
                                    "}").replace("], [", ", ").replace("\n", "").replace("\\n", "").replace("\\",
                                     "").replace('""', '"').replace('，', ',')
    answer = replace_double_quota(answer)
    answer = answer.replace("{{", "{").replace("}}", 
                                    "}").replace("], [", ", ").replace("\n", "").replace("\\n", "").replace("\\",
                                     "").replace('""', '"').replace('，', ',')


    # annotation = annotation.replace("\'", "\"")
    # answer = answer.replace("\'", "\"")

    annotation_list = json.loads(annotation)
    annotation_list = [item["locate"] for item in annotation_list]

    answer_list = json.loads(answer)
    answer_list = [item["locate"] for item in answer_list]
    return jaccard_similarity(annotation_list, answer_list)

def language_score_cul(predict, bench, tag, i):
    predict_answer = get_between_multi(predict['predict'], '<answer>', '</answer>')
    bench_answer = get_between_strings_regex(bench['output'], '<answer>', '</answer>')
    atype = str(tag['atype'])
    ttype = tag['table_name']
    language = tag['language']
    score = 0
    # print(atype)
    # print(len(predict_answer))
    for pre_answer in predict_answer:
        if atype == "1":
            # 数值计算
            try:
                digit_score = get_digit_score(pre_answer, bench_answer)
                score=score if score > digit_score else digit_score
                # print('1')
            except Exception as e:
                # print(e)
                pass
                # print(i,pre_answer,'——————', bench_answer, '——————', bench['output'])

        elif atype == "2":
            # 单元格定位
            try:
                locate_score=get_locate_scores(pre_answer, bench_answer)
                score=score if score > locate_score else locate_score
            except Exception as e:
                pass
                # print(i,pre_answer,'——————', bench_answer, '——————', bench['output'])
        elif atype == "3":
            # 对错
            true_false_scores = 1 if pre_answer.lower() == bench_answer.lower() else 0
            score=score if score > true_false_scores else true_false_scores
        elif atype == "4":
            rouge = Rouge()
            # print(pre_answer)
            # print(bench_answer)
            try:
                scores = rouge.get_scores(pre_answer, bench_answer, avg=True)
                # print(i,scores['rouge-1']['f'])
                # print(scores)
                score=score if score > scores['rouge-l']['f'] else scores['rouge-l']['f']
            except Exception as e:
                pass
                # print(i,pre_answer,'——————', bench_answer, '——————', bench['output'])
    return score, atype, ttype

def get_language_scores(predict_data, bench_data, tag_data):
    i = 0
    total_digit_scores = []
    total_locate_scores = []
    total_true_false_scores = []
    total_nl_scores = []
    language_scores = {}
    bench_data = bench_data[:len(predict_data)]
    tag_data = tag_data[:len(predict_data)]
    for predict, bench, tag in zip(predict_data, bench_data, tag_data):
        i += 1
        # if i == 2094:
        score, atype, ttype = language_score_cul(predict, bench, tag, i)
        if atype == '1':
            total_digit_scores.append(score)
        elif atype == '2':
            total_locate_scores.append(score)
        elif atype == '3':
            total_true_false_scores.append(score)
        elif atype == '4':
            total_nl_scores.append(score)

        
        language = tag['language']
        if language in language_scores:
            lang_score = language_scores[language]
            if atype == '1':
                language_scores[language]['digit_scores'].append(score)
            elif atype == '2':
                language_scores[language]['locate_scores'].append(score)
            elif atype == '3':
                language_scores[language]['true_false_scores'].append(score)
            elif atype == '4':
                language_scores[language]['nl_scores'].append(score)
        else:
            language_scores[language] = {}
            language_scores[language]['digit_scores'] = []
            language_scores[language]['locate_scores'] = []
            language_scores[language]['true_false_scores'] = []
            language_scores[language]['nl_scores'] = []
            if atype == '1':
                digit_scores=[score]
                language_scores[language]['digit_scores'] = digit_scores
            elif atype == '2':
                locate_scores=[score]
                language_scores[language]['locate_scores'] = locate_scores
            elif atype == '3':
                true_false_scores=[score]
                language_scores[language]['true_false_scores'] = true_false_scores
            elif atype == '4':
                nl_scores=[score]
                language_scores[language]['nl_scores'] = nl_scores

    print(sum(total_digit_scores)+sum(total_locate_scores)+sum(total_true_false_scores)+sum(total_nl_scores))
    print(len(total_digit_scores)+len(total_locate_scores)+len(total_true_false_scores)+len(total_nl_scores))
    print("数值型答案得分：" + str(sum(total_digit_scores) / len(total_digit_scores)))
    print("定位型答案得分：" + str(sum(total_locate_scores) / len(total_locate_scores)))
    print("对错型答案得分：" + str(sum(total_true_false_scores) / len(total_true_false_scores)))
    print("问答型答案得分：" + str(sum(total_nl_scores) / len(total_nl_scores)))
    print("总得分：" + str((sum(total_digit_scores)+sum(total_locate_scores)+sum(total_true_false_scores)+sum(total_nl_scores)) / (len(total_digit_scores)+len(total_locate_scores)+len(total_true_false_scores)+len(total_nl_scores))))

    return language_scores

def get_table_type_scores(predict_data, bench_data, tag_data):
    i = 0
    total_1_scores = []
    total_2_scores = []
    total_3_scores = []
    total_4_scores = []
    language_scores = {}
    bench_data = bench_data[:len(predict_data)]
    tag_data = tag_data[:len(predict_data)]
    for predict, bench, tag in zip(predict_data, bench_data, tag_data):
        i += 1
        # if i == 2094:
        score, atype, ttype = language_score_cul(predict, bench, tag, i)
        if ttype.startswith("关系"):
            total_1_scores.append(score)
        elif ttype.startswith("实体"):
            total_2_scores.append(score)
        elif ttype.startswith("矩阵"):
            total_3_scores.append(score)
        elif ttype.startswith("复合"):
            total_4_scores.append(score)
        elif ttype == "补充1":
            total_4_scores.append(score)
        else:
            total_1_scores.append(score)

        
        language = tag['language']
        if language in language_scores:
            lang_score = language_scores[language]
            if atype == '1':
                language_scores[language]['digit_scores'].append(score)
            elif atype == '2':
                language_scores[language]['locate_scores'].append(score)
            elif atype == '3':
                language_scores[language]['true_false_scores'].append(score)
            elif atype == '4':
                language_scores[language]['nl_scores'].append(score)
        else:
            language_scores[language] = {}
            language_scores[language]['digit_scores'] = []
            language_scores[language]['locate_scores'] = []
            language_scores[language]['true_false_scores'] = []
            language_scores[language]['nl_scores'] = []
            if atype == '1':
                digit_scores=[score]
                language_scores[language]['digit_scores'] = digit_scores
            elif atype == '2':
                locate_scores=[score]
                language_scores[language]['locate_scores'] = locate_scores
            elif atype == '3':
                true_false_scores=[score]
                language_scores[language]['true_false_scores'] = true_false_scores
            elif atype == '4':
                nl_scores=[score]
                language_scores[language]['nl_scores'] = nl_scores

    print(sum(total_1_scores)+sum(total_2_scores)+sum(total_3_scores)+sum(total_4_scores))
    print(len(total_1_scores)+len(total_2_scores)+len(total_3_scores)+len(total_4_scores))
    print("关系表格得分：" + str(sum(total_1_scores) / len(total_1_scores)))
    print("实体表格得分：" + str(sum(total_2_scores) / len(total_2_scores)))
    print("矩阵表格得分：" + str(sum(total_3_scores) / len(total_3_scores)))
    print("复合表格得分：" + str(sum(total_4_scores) / len(total_4_scores)))
    print("总得分：" + str((sum(total_1_scores)+sum(total_2_scores)+sum(total_3_scores)+sum(total_4_scores)) / (len(total_1_scores)+len(total_2_scores)+len(total_3_scores)+len(total_4_scores))))

    return language_scores

def main():
    
    question_type_count = {}
    answer_type_count = {}
    
    write_result = True

    output_path = "/volume/pt-train/users/wzhang/sdx-workspace/output/result"

    predict_artificial_path = "/volume/pt-train/users/wzhang/sdx-workspace/output/LLM-Research/Meta-Llama-3.1-8B-Instruct-artificial-data/sft_think/generated_predictions.jsonl"
    predict_llm_path = "/volume/pt-train/users/wzhang/sdx-workspace/output/LLM-Research/Meta-Llama-3.1-8B-Instruct/sft_think/generated_predictions.jsonl"


    llm_tag_path = "/volume/pt-train/users/wzhang/sdx-workspace/LLaMA-Factory/data/mmt_test_full_tags.json"
    artificial_tag_path = "/volume/pt-train/users/wzhang/sdx-workspace/LLaMA-Factory/data/mmt_test_artificial_no_think_tags.json"

    test_artificial_type = "mmt_test_artificial_no_think"
    test_llm_type = "mmt_test_no_think"
    dataset_base_path = '/volume/pt-train/users/wzhang/sdx-workspace/LLaMA-Factory/data'
    with open(os.path.join(dataset_base_path, 'dataset_info.json'), 'r')as f:
        dataset_info = json.load(f)
    test_artificial_name = dataset_info[test_artificial_type]['file_name']
    test_llm_name = dataset_info[test_llm_type]['file_name']

    bench_artificial_path = os.path.join(dataset_base_path, test_artificial_name)
    bench_llm_path = os.path.join(dataset_base_path, test_llm_name)

    tag_artificial_path = artificial_tag_path
    tag_llm_path = llm_tag_path

    predict_artificial_data = []
    with open(predict_artificial_path, 'r') as file:
        for line in file:
            predict_artificial_data.append(json.loads(line))
            
    predict_llm_data = []
    with open(predict_llm_path, 'r') as file:
        for line in file:
            predict_llm_data.append(json.loads(line))
    
    bench_artificial_data = None
    with open(bench_artificial_path, 'r') as file:
        bench_artificial_data = json.load(file)
        
    bench_llm_data = None
    with open(bench_llm_path, 'r') as file:
        bench_llm_data = json.load(file)
    
    tag_artificial_data = None
    with open(tag_artificial_path, 'r') as file:
        tag_artificial_data = json.load(file)
        
    tag_llm_data = None
    with open(tag_llm_path, 'r') as file:
        tag_llm_data = json.load(file)

    print("人工构造数据得分：")
    language_scores_artificial = get_language_scores(predict_artificial_data, bench_artificial_data, tag_artificial_data)
    print(language_scores_artificial.keys())
    print("LLM生成数据得分：")
    language_scores_llm = get_language_scores(predict_llm_data, bench_llm_data, tag_llm_data)

    total_predict = predict_llm_data
    total_predict.extend(predict_artificial_data)
    total_bench = bench_llm_data
    total_bench.extend(bench_artificial_data)
    total_tag = tag_llm_data
    total_tag.extend(tag_artificial_data)
    print(len(total_predict), len(total_bench), len(total_tag))
    print("总得分：")
    total_scores = get_language_scores(total_predict, total_bench, total_tag)
    _ = get_table_type_scores(total_predict, total_bench, total_tag)

    language_scores = language_scores_llm
    for lang in language_scores:
        lang_score = language_scores[lang]
        if lang in language_scores_artificial:
            lang_score_art = language_scores_artificial[lang]
            lang_score['digit_scores'].extend(lang_score_art['digit_scores'])
            lang_score['locate_scores'].extend(lang_score_art['locate_scores'])
            lang_score['true_false_scores'].extend(lang_score_art['true_false_scores'])
            lang_score['nl_scores'].extend(lang_score_art['nl_scores'])
    
    # 按表格类型计算得分
    # table_type_data = []
    # for lang in language_scores:

    # 按答案类型计算得分
    output_data = []
    for lang in language_scores:
        # print(lang)
        lang_score = language_scores[lang]
        try:
            digit_mean = str(sum(lang_score['digit_scores']) / len(lang_score['digit_scores']))
        except:
            digit_mean = '/'
        try:
            locate_mean = str(sum(lang_score['locate_scores']) / len(lang_score['locate_scores']))
        except:
            locate_mean = '/'
        try:
            true_false_mean = str(sum(lang_score['true_false_scores']) / len(lang_score['true_false_scores']))
        except:
            true_false_mean = '/'
        try:
            nl_mean = str(sum(lang_score['nl_scores']) / len(lang_score['nl_scores']))
        except:
            nl_mean = '/'
        try:
            total_sum = sum(lang_score['digit_scores'])+sum(lang_score['locate_scores'])+sum(lang_score['true_false_scores'])+sum(lang_score['nl_scores'])
            total_len = len(lang_score['digit_scores'])+len(lang_score['locate_scores'])+len(lang_score['true_false_scores'])+len(lang_score['nl_scores'])
            total_mean = str(total_sum / total_len)
        except:
            total_mean = '/'
        output_data.append([lang, digit_mean, locate_mean, true_false_mean, nl_mean, total_mean])

    if write_result:
        with open(os.path.join(output_path, 'atype_result.csv'), 'w', newline='')as f:
            writer = csv.writer(f)
            writer.writerows(output_data)

    # 按语系计算得分
    language_family_path = '/volume/pt-train/users/wzhang/sdx-workspace/LLaMA-Factory/config/language_family.json'
    language_family_score = {}
    with open(language_family_path, 'r')as f:
        language_family = json.load(f)
    # print(language_family)
    for lang in language_scores:
        lang_score = language_scores[lang]
        for lf in language_family:
            if lang in language_family[lf]:
                # print(lf)
                # flag = 1
                if lf in language_family_score:
                    language_family_score[lf]['digit_scores'].extend(lang_score['digit_scores'])
                    language_family_score[lf]['locate_scores'].extend(lang_score['locate_scores'])
                    language_family_score[lf]['true_false_scores'].extend(lang_score['true_false_scores'])
                    language_family_score[lf]['nl_scores'].extend(lang_score['nl_scores'])
                else:
                    lf_score = {
                        "digit_scores": lang_score['digit_scores'] if lang_score['digit_scores'] != None else [], 
                        "locate_scores": lang_score['locate_scores'] if lang_score['locate_scores'] != None else [],
                        "true_false_scores": lang_score['true_false_scores'] if lang_score['true_false_scores'] != None else [],
                        "nl_scores": lang_score['nl_scores'] if lang_score['nl_scores'] != None else []
                    }
                    language_family_score[lf] = lf_score
                break
    language_family_result = []
    for lf in language_family_score:
        lf_score = language_family_score[lf]
        try:
            digit_mean = str(round(sum(lf_score['digit_scores']) / len(lf_score['digit_scores'])*100, 2))
        except:
            digit_mean = '/'
        try:
            locate_mean = str(round(sum(lf_score['locate_scores']) / len(lf_score['locate_scores'])*100, 2))
        except:
            locate_mean = '/'
        try:
            true_false_mean = str(round(sum(lf_score['true_false_scores']) / len(lf_score['true_false_scores'])*100, 2))
        except:
            true_false_mean = '/'
        try:
            nl_mean = str(round(sum(lf_score['nl_scores']) / len(lf_score['nl_scores'])*100, 2))
        except:
            nl_mean = '/'
        try:
            total_sum = sum(lf_score['digit_scores'])+sum(lf_score['locate_scores'])+sum(lf_score['true_false_scores'])+sum(lf_score['nl_scores'])
            total_len = len(lf_score['digit_scores'])+len(lf_score['locate_scores'])+len(lf_score['true_false_scores'])+len(lf_score['nl_scores'])
            total_mean = str(round(total_sum / total_len*100, 2))
        except:
            total_mean = '/'
        language_family_result.append([lf, digit_mean, locate_mean, true_false_mean, nl_mean, total_mean])

    
    if write_result:
        with open(os.path.join(output_path, 'language_family_result.csv'), 'w', newline='')as f:
            writer = csv.writer(f)
            writer.writerows(language_family_result)

        # print(language_family_result[0][1:])
        order_result = language_family_result[0][1:]
        order_result.extend(language_family_result[1][1:])
        order_result.extend(language_family_result[2][1:])
        order_result.extend(language_family_result[4][1:])
        order_result.extend(language_family_result[5][1:])
        order_result.extend(language_family_result[6][1:])
        order_result.extend(language_family_result[7][1:])
        order_result.extend(language_family_result[8][1:])
        order_result.extend(language_family_result[9][1:])
        order_result.extend(language_family_result[10][1:])
        order_result.extend(language_family_result[11][1:])
        order_result.extend(language_family_result[3][1:])
        # print(order_result)
        with open(os.path.join(output_path, 'order_result.csv'), 'w', newline='')as f:
            writer = csv.writer(f)
            writer.writerows([order_result])

if __name__ == '__main__':
    main()
