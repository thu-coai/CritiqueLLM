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

base_prompt = "你是一个擅长评价文本质量的助手。\n请你以公正的评判者的身份，比较两个AI助手对于用户提问的回答的质量优劣。" \
            "由于你评估的任务类型是{category}，因此你需要从下面的几个维度对两个回答进行细致的比较和分析:\n{dimensions}" \
            "我们会给你提供用户的提问，需要你比较的两个AI助手的答案，以及两个答案各自的质量评价分析。" \
            "当你开始你的评估时，你需要遵守以下的流程：\n" \
            "1. 结合两个AI助手的答案以及其质量评价分析，根据上述指定的维度对他们的答案进行细致的比较，给出详细的比较分析文本。" \
            "比较分析文本要求覆盖两个答案的质量评价分析中可用于比较的所有重要细节，并包含对答案中具体内容的分析。\n" \
            "2. 结合每个维度的比较分析，从两个AI助手的答案中选出综合质量更高的那个，或者判定他们质量相当，并给出详尽的选择理由。\n" \
            "你的比较需要尽可能严谨细致，不受两个AI助手答案先后顺序的影响。\n" \
            "质量评价分析中的各维度分数和综合得分仅供参考，在各维度和综合的比较分析文本中不能直接提及各维度分数和综合得分。针对综合得分" \
            "差距较大的样本对，应尽可能按照分数高低得出比较结果，除非发现质量评价" \
            "分析中存在明显错误。而针对综合得分差距较小的样本对，则允许比较结果和分数高低不一致，但仍需要详细说明比较评价的理由。\n"  \
            "请记住，你必须首先按照给定的评价维度，输出相应维度的名称和比较分析的文本。然后再给出综合质量比较结果，并给出比较结果" \
            "的分析和解释。之后，在你回答的末尾，按照以下字典格式（包括括号）返回你的综合质量选择结果，即你选择的综合质量更高的那个AI助手" \
            "（或者认为质量相当），并确保你返回的结果和上述生成文本中的结果保持一致：\n" \
            "{{'综合比较结果': 回答综合质量更高的助手序号或质量相当}}，例如：{{'综合比较结果': '助手1'}}或{{'综合比较结果': '助手2'}}或" \
            "{{'综合比较结果': '质量相当'}}。\n" \
            "用户的提问：{question}\n" \
            "[助手1的答案开始]\n{response_1}\n[助手1的答案结束]\n" \
            "[助手1的答案质量评价分析开始]\n{pointwise_reference_free_judgement_1}\n[助手1的答案质量评价分析结束]\n" \
            "[助手2的答案开始]\n{response_2}\n[助手2的答案结束]\n" \
            "[助手2的答案质量评价分析开始]\n{pointwise_reference_free_judgement_2}\n[助手2的答案质量评价分析结束]\n"


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
                data['pointwise_reference_free_to_pairwise_prompt'] = payload['messages'][0]['content']
                data['pointwise_reference_free_to_pairwise_judgement'] = chat_completion.choices[0].message.content
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
                                    response_1=d['response_1'],
                                    response_2=d['response_2'],
                                    pointwise_reference_free_judgement_1=d['pointwise_reference_free_judgement_1'],
                                    pointwise_reference_free_judgement_2=d['pointwise_reference_free_judgement_2'])
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
        d['pointwise_reference_free_to_pairwise_prompt'] = results[i]['pointwise_reference_free_to_pairwise_prompt']
        d['pointwise_reference_free_to_pairwise_judgement'] = results[i]['pointwise_reference_free_to_pairwise_judgement']
        outs.append(d)
    
    with open(args.output_path, "w", encoding='utf-8') as f:
        for d in outs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")