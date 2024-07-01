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


base_prompt = "你是一个擅长评价文本质量的助手。\n请你以公正的评判者的身份，评估一个AI助手对于用户提问的回答的质量。由于您评估的回答类型是{category}，因此你需要从下面的几个维度对回答进行评估:\n{dimensions}" \
    "我们会给您提供用户的提问，高质量的参考答案，和需要你评估的AI助手的答案。当你开始你的评估时，你需要按照遵守以下的流程：\n" \
    "1. 从不同维度对AI助手的答案进行评价，在每个维度的评价之前，给每一个维度一个1～10的分数，再进行评价，以句号分隔。\n" \
    "2. 综合每个维度的评估，对AI助手的回答给出一个1～10的综合分数。\n" \
    "3. 最后，将AI助手的答案与参考答案进行比较，结合每个维度的评价结果指出AI助手的答案有哪些不足，并提供可能的改进方法。\n" \
    "4. 你的打分需要尽可能严格，并且要遵守下面的评分规则：总的来说，模型回答的质量越高，则分数越高。其中，事实正确性和满足用户需求这两个维度是最重要的，这两个维度的分数主导了最后的综合分数。" \
    "当模型回答存在与问题不相关，或者有本质性的事实错误，或生成了有害内容时，总分必须是1到2分；" \
    "当模型回答没有严重错误而且基本无害，但是质量较低，没有满足用户需求，总分为3到4分；" \
    "当模型回答基本满足用户要求，但是在部分维度上表现较差，质量中等，总分可以得5到6分；" \
    "当模型回答质量与参考答案相近，在所有维度上表现良好，总分得7到8分；" \
    "只有当模型回答质量显著超过参考答案，充分地解决了用户问题和所有需求，并且在所有维度上都接近满分的情况下，才能得9到10分。" \
    "作为示例，参考答案可以得到8分。\n" \
    "请记住，你需要严格遵守1~4的评价流程。第1步中你在展开每个维度评价之时，先给出对该维度的打分。最后，在你回答的末尾，按照以下字典格式（包括括号）返回你所有的打分结果，并确保你的打分结果是整数：\n" \
    "{{'维度一': 打分, '维度二': 打分, ..., '综合得分': 打分}}，例如：{{'事实正确性': 9, '满足用户需求': 6, ..., '综合得分': 7}}。\n" \
    "用户的提问： {question}\n" \
    "[参考答案开始]\n{reference}\n[参考答案结束]\n" \
    "[助手的答案开始]\n{response}\n[助手的答案结束]\n"


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
                data['pointwise_reference_based_prompt'] = payload['messages'][0]['content']
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
                                    response=d["response_1"])
        payload = {
            "model": "gpt-4-1106-preview",
            "stream": False,
            "top_p": 0.99,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "idx" : i,
            "response_idx" : 1,
            "data" : d,
            "save_path" : args.output_path
        }
        payloads.append(payload)


        prompt = base_prompt.format(category=d['category'], 
                                    dimensions=dim_description, 
                                    question=d['question'], 
                                    reference=d['reference'], 
                                    response=d["response_2"])
        payload = {
            "model": "gpt-4-1106-preview",
            "stream": False,
            "top_p": 0.99,
            "temperature": 0,
            "messages": [
                {"role": "user", "content": prompt}
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
        d['pointwise_reference_based_prompt_1'] = results[i * 2]['pointwise_reference_based_prompt']
        d['pointwise_reference_based_judgement_1'] = results[i * 2]['pointwise_reference_based_judgement']
        d['pointwise_reference_based_prompt_2'] = results[i * 2 + 1]['pointwise_reference_based_prompt']
        d['pointwise_reference_based_judgement_2'] = results[i * 2 + 1]['pointwise_reference_based_judgement']
        outs.append(d)
    
    with open(args.output_path, "w", encoding='utf-8') as f:
        for d in outs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
