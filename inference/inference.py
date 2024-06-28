import json
import os
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from copy import deepcopy
from prompts import *

def read_data(args):
    with open(args.maps, "r", encoding='utf-8') as f:
        maps = json.load(f)
    
    data = []
    with open(args.input_path, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    
    for i in range(len(data)):
        if args.pointwise:
            if args.reference_free:
                data[i]["input"] = Pointwise_reference_free.format(type=maps[data[i]['category']], question=data[i]['question'], response=data[i]['response'])
            else:
                data[i]["input"] = Pointwise_reference_based.format(type=maps[data[i]['category']], question=data[i]['question'], reference=data[i]['reference'], response=data[i]['response'])
        else:
            if args.reference_free:
                data[i]["input"] = Pairwise_reference_free.format(type=maps[data[i]['category']], question=data[i]['question'], response_1=data[i]['response_1'], response_2=data[i]['response_2'])
            else:
                data[i]["input"] = Pairwise_reference_based.format(type=maps[data[i]['category']], question=data[i]['question'], reference=data[i]['reference'], response_1=data[i]['response_1'], response_2=data[i]['response_2'])

    data = sorted(data, key = lambda x : len(x["input"]))
    return data


def batch_generate(model, tokenizer, save_path, data, batch_size, args):
    length = (len(data) - 1) // batch_size + 1
    with open(save_path, "w", encoding='utf-8') as f:
        for i in tqdm(range(length)):
            if (i + 1) * batch_size <= len(data):
                texts = data[i * batch_size : (i + 1) * batch_size]
            else:
                texts = data[i * batch_size :]
            
            batch_texts = [x["input"] for x in texts]
            encoding = tokenizer(batch_texts, padding=True, return_tensors='pt').to('cuda')
            
            generation_args = dict(
                do_sample=args.do_sample,
                top_p=0.9,
                temperature=0.9,
                num_beams=1,
                length_penalty=1.0,
                max_length=encoding['input_ids'][0].shape[0] + 1024
            )
            
            generated_ids = model.generate(**encoding, **generation_args)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i in range(len(texts)):
                o = {}
                o = deepcopy(texts[i])
                o['output'] = generated_texts[i]
                json.dump(o, f, ensure_ascii=False)
                f.write("\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="THUDM/chatglm3-6b")
    parser.add_argument('--input_path', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--pointwise', type=bool, default=False)
    parser.add_argument('--reference_free', type=bool, default=False)
    parser.add_argument('--maps', type=str, default=False)
    args = parser.parse_args()

    batch_size = 24

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()
    tokenizer.padding_side = "left" 
    print("Load Model Done")
    print("Model Path", args.model_path)
    
    model_name = args.model_path
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    if model_name[0] == "/":
        model_name = model_name[1:]
    model_name = model_name.split("/")[-1]
    save_path = os.path.join(args.output_path, model_name)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    data = read_data(args)
    print("Generate Begin!")
    batch_generate(model, tokenizer, os.path.join(save_path, args.output_name), data, args.batch_size, args)
        