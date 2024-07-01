import json
import random
import time
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import openai
import argparse
import traceback
from copy import deepcopy


base_prompt = "你是一个擅长评价文本质量的助手。\n请你根据以下要求修改比较式评价文本。\n" \
            "1. 在修改后的评价文本中，不要直接提及参考答案。可以在评价文本中适当利用参考答案中的具体内容辅助分析，但不要让读者感受到参考答案的存在。" \
            "修改后的评价文本需要语言上通顺，逻辑上合理，分析内容与比较结果呼应。\n" \
            "2. 在修改各个维度的比较分析时，分析的内容需要和当前评价文本基本保持一致，但不要直接提及参考答案。\n" \
            "3. 在修改综合比较结果的分析文本时，不要直接提及参考答案，尽量保留当前评价文本中的其他细节，并充分利用修改后的" \
            "分维度分析。修改后的综合分析文本应通顺、流畅、自洽，通常情况下应与综合比较结果保持一致。如果发现当前综合分析文本中存在重要错误，" \
            "应修改相应的分析文本。仅当该错误严重影响到综合比较结果时，才慎重修改综合比较结果。\n" \
            "4. 修改后所有输出格式需要和当前评价文本严格保持一致。在你回答的末尾，仍需要按照以下字典格式（包括括号）返回你的综合质量" \
            "选择结果，即你选择的综合质量更高的那个AI助手（或者认为质量相当），并确保你返回的结果和上述生成文本中的结果保持一致：\n" \
            "{{'综合比较结果': 回答综合质量更高的助手序号或质量相当}}，例如：{{'综合比较结果': '助手1'}}或{{'综合比较结果': '助手2'}}或" \
            "{{'综合比较结果': '质量相当'}}。\n" \
            "用户的提问：{question}\n" \
            "[参考答案开始]\n{reference}\n[参考答案结束]\n" \
            "[助手1的答案开始]\n{response_1}\n[助手1的答案结束]\n" \
            "[助手2的答案开始]\n{response_2}\n[助手2的答案结束]\n" \
            "[评价文本开始]\n{pairwise_reference_based_judgement}\n[评价文本结束]\n"


class ChatGPTProcessor:
    def __init__(self, args):
        self.lock = multiprocessing.Lock()
        openai.api_key = args.api_key
        openai.api_base = args.api_base

    def gpt(self, payload):
        while True:
            try:
                chat_completion = openai.ChatCompletion.create(model=payload['model'], temperature=0, messages=payload['messages'])
                data = deepcopy(payload['data'])
                data['idx'] = payload['idx']
                data['pairwise_reference_based_to_reference_free_prompt'] = payload['messages'][0]['content']
                data['pointwise_reference_based_judgement'] = chat_completion.choices[0].message.content
                break
            except Exception as e:
                traceback.print_exc()
                time.sleep(random.randint(1, 3))

        time.sleep(random.randint(1, 3))
        return data    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--api_base', type=str, default="")
    parser.add_argument('--api_key', type=str, default="")
    args = parser.parse_args()
    processor = ChatGPTProcessor(args)

    with open(args.input_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    with open("config/dimension_definition.json", "r", encoding='utf-8') as f:
        dimension_def_map = json.load(f)
    with open("config/types.json", "r", encoding='utf-8') as f:
        types = json.load(f)
    with open("config/category2type.json", "r", encoding='utf-8') as f:
        category2type = json.load(f)
    
    payloads = []
    for i, d in enumerate(data):
        dimensions = types[category2type[d['category']]]
        dim_description = ""
        for index, dim in enumerate(dimensions):
            dim_description += f"{index+1}. {dim}: {dimension_def_map[dim]}\n"
        prompt = base_prompt.format(category=d['category'], 
                                    dimensions=dim_description, 
                                    question=d['question'], 
                                    reference=d['reference'], 
                                    response_1=d['response_1'],
                                    response_2=d['response_2'],
                                    pairwise_reference_based_judgement=d['pairwise_reference_based_judgement'])
        payload = {
            "model": "gpt-4-1106-preview",
            "stream": False,
            "top_p": 0.99,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "idx" : i,
            "data" : d,
            "save_path" : args.output_path
        }
        payloads.append(payload)


    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(tqdm(executor.map(processor.gpt, payloads), total=len(payloads), desc='Processing', ncols=100))
    
    print("Done!")
    results.sort(key=lambda x : x['idx'])

    outs = []
    for i, d in enumerate(data):
        d['pairwise_reference_based_to_reference_free_prompt'] = results[i]['pairwise_reference_based_to_reference_free_prompt']
        d['pairwise_reference_based_to_reference_free_judgement'] = results[i]['pairwise_reference_based_to_reference_free_judgement']
        outs.append(d)
    
    with open(args.output_path, "w", encoding='utf-8') as f:
        for d in outs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
