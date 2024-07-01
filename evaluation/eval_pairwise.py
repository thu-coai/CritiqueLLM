import json

def extract_label(s):
    if "{'综合比较结果': '助手1'}" in s:
        return 0
    if "{'综合比较结果': '助手2'}" in s:
        return 1
    if "{'综合比较结果': '质量相当'}" in s:
        return 2
    return -1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    args = parser.parse_args()

    labels = []
    with open(args.input_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]    
    data.sort(key=lambda x: x['id'])
    scores = []
    for d in data:
        scores.append(extract_label(d['output']))
        labels.append(d['label'])

    con = 0
    agr = 0 
    for i in range(len(scores)):
        if i % 2 == 1:
            continue
        
        if scores[i] == 0 and scores[i + 1] == 1:
            con += 1
        if scores[i] == 1 and scores[i + 1] == 0:
            con += 1
        if scores[i] == 2 and scores[i + 1] == 2:
            con += 1

        if scores[i] == 0 and scores[i + 1] == 1 and labels[i] == 0:
            agr += 1
        elif scores[i] == 1 and scores[i + 1] == 0 and labels[i] == 1:
            agr += 1
        elif scores[i] == 2 and scores[i + 1] == 2 and labels[i] == 2:
            agr += 1
        
    
    result = {}
    result['agreement'] = agr * 2 / len(scores)
    result['consistency'] = con * 2 / len(scores)
    
    with open(args.output_path + "/result.json", "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    