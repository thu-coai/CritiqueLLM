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


ref_free_prompts = "请根据以下要求修改你上一轮给出的评价分数和解释：\n" \
    "1. 在修改后的评价结果中，不要直接提及参考答案。修改后的评价需要语言上通顺，逻辑上合理，与打分呼应。\n" \
    "2. 在修改各个维度的评价时，保持先给出分数，再进行评价的顺序。各个维度的分数和综合分数相比于上一轮允许有不超过1分的改动。评价的内容需要和上一轮的保持一致，但禁止提及参考答案。\n" \
    "3. 在修改综合评价时，你需要先给出分数，然后评价，指出AI助手的答案有哪些不足，并提供可能的改进方法。上一轮你通过与参考答案比较的方式指出了AI助手答案的不足，而现在你不准提及参考答案，但你可利用上一轮评价中的细节，指出不足与提供AI助手的改进方法。修改后的评价、改进方法结果应通顺、流畅、自洽。\n" \
    "4. 修改后所有输出格式需要和上一轮的评价结果严格保持一致，你需按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：\n" \
    "{{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}}，例如：{{'事实正确性': 9, '满足用户需求': 6, ..., '综合得分': 7}}。\n"

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
                data['response_idx'] = payload['response_idx']
                data['pointwise_reference_free_judgement'] = chat_completion.choices[0].message.content
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
    
    payloads = []
    for i, d in enumerate(data):
        payload = {
            "model": "gpt-4-1106-preview",
            "stream": False,
            "top_p": 0.99,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": d['pointwise_reference_based_prompt_1']},
                {"role": "assistant", "content": d['pointwise_reference_based_judgement_1']},
                {"role": "user", "content": ref_free_prompts}
            ],
            "idx" : i,
            "response_idx" : 1,
            "data" : d,
            "save_path" : args.output_path
        }
        payloads.append(payload)

        payload = {
            "model": "gpt-4-1106-preview",
            "stream": False,
            "top_p": 0.99,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": d['pointwise_reference_based_prompt_2']},
                {"role": "assistant", "content": d['pointwise_reference_based_judgement_2']},
                {"role": "user", "content": ref_free_prompts}
            ],
            "idx" : i,
            "response_idx" : 2,
            "data" : d,
            "save_path" : args.output_path
        }
        payloads.append(payload)

    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(tqdm(executor.map(processor.gpt, payloads), total=len(payloads), desc='Processing', ncols=100))
    
    print("Done!")
    results.sort(key=lambda x : (x['idx'], x['response_idx']))

    outs = []
    for i, d in enumerate(data):
        d['pointwise_reference_free_judgement_1'] = results[i * 2]['pointwise_reference_free_judgement']
        d['pointwise_reference_free_judgement_2'] = results[i * 2 + 1]['pointwise_reference_free_judgement']
        outs.append(d)
    
    with open(args.output_path, "w", encoding='utf-8') as f:
        for d in outs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")