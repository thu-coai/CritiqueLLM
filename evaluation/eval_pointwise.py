import json
import re
import numpy as np
import os
import argparse
import scipy.stats
from scipy.stats import spearmanr, kendalltau

def spearman_corr(x, y):
    corr, _ = spearmanr(x, y)
    return corr


def kendall_corr(x, y):
    corr, _ = kendalltau(x, y)
    return corr


def extract_number(s):
    pattern_1 = r'\'综合得分\': ((?:0|10)(?:\.0*)?|[0-9](?:\.\d*)?)'
    match_1 = re.search(pattern_1, s)
    if match_1:
        return float(match_1.group(1))

    pattern_2 = r'综合得分: ((?:0|10)(?:\.0*)?|[0-9](?:\.\d*)?)'
    match_2 = re.search(pattern_2, s)
    if match_2:
        return float(match_2.group(1))
    
    pattern_3 = r'综合得分：((?:0|10)(?:\.0*)?|[0-9](?:\.\d*)?)'
    match_3 = re.search(pattern_3, s)
    if match_3:
        return float(match_3.group(1))

    return 5


def correlation(v, u):
    return {
        "Pearson" : np.corrcoef(v, u)[0, 1],
        "Spearman" : spearman_corr(v, u),
        "Kendall" : kendall_corr(v, u)
    }


def correlation_sample(v, u):
    avg_p = 0
    avg_s = 0
    avg_k = 0
    i = 0
    cnt = 0
    while i + 8 <= len(v):
        vs = v[i: i + 8]
        us = u[i: i + 8]
        if np.isnan(scipy.stats.spearmanr(vs, us)[0]):
            i += 8
            continue
        avg_p += np.corrcoef(vs, us)[0, 1]
        avg_s += scipy.stats.spearmanr(vs, us)[0]
        avg_k += scipy.stats.kendalltau(vs, us)[0]
        cnt += 1
        i += 8
    
    avg_p /= cnt
    avg_s /= cnt
    avg_k /= cnt
    return {"pearson": avg_p, "spearman": avg_s, "kendall": avg_k}


def correlation_system_level(v, u, human_data):
    keys = {}
    vs = {}
    us = {}
    for i, x in enumerate(human_data):
        if x['model'] not in keys.keys():
            keys[x['model']] = []
            vs[x['model']] = []
            us[x['model']] = []
        
        vs[x['model']].append(v[i])
        us[x['model']].append(u[i])
    
    vm = []
    _vm = []
    um = []
    _um = []
    for x in keys.keys():
        _vm.append((x, np.mean(vs[x])))
        _um.append((x, np.mean(us[x])))
        vm.append(np.mean(vs[x]))
        um.append(np.mean(us[x]))
    
    return {
        "human_score" : _vm,
        "model_score" : _um,
        "system_level_corr" : correlation(vm, um)
    }


def correlation_per_category(v, u, human_data):
    correlations = {}
    vs = {}
    us = {}
    for i, x in enumerate(human_data):
        if x['category'] not in correlations.keys():
            correlations[x['category']] = []
            vs[x['category']] = []
            us[x['category']] = []
        
        vs[x['category']].append(v[i])
        us[x['category']].append(u[i])
    
    for x in correlations.keys():
        correlations[x] = correlation_sample(vs[x], us[x])
    return correlations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    data.sort(key=lambda x: x['id'])

    critiquellm_scores = []
    human_scores = []
    human_data = []
    for i, d in enumerate(data):
        input = d['input']
        output = d['output']
        output = output.replace(input, "")
        score = extract_number(output)
        if score == None:
            score = 5                
        
        critiquellm_scores.append(score)
        human_scores.append(d['human_score'])
        human_data.append(d)
                    
    critiquellm_scores = np.array(critiquellm_scores)
    human_scores = np.array(human_scores)
    result = {}
    result['human_correlation_sample_level'] = correlation_sample(human_scores, critiquellm_scores)
    result['human_correlation_sample_level_per_category'] = correlation_per_category(human_scores, critiquellm_scores, human_data)
    result['human_correlation_system_level'] = correlation_system_level(human_scores, critiquellm_scores, human_data)
    with open(args.output_path + "/result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
