# CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation

This repository is the official implementation of [CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation](https://arxiv.org/abs/2311.18702). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Download

You can download CritiqueLLM-6B [here](https://huggingface.co/thu-coai/CritiqueLLM-6B).

## Data Collection

Some samples of training data are placed in `data/train`.  The format of training data is as follows:

- `id` (integer): A unique identifier for the instance.
- `question` (string): The actual user query. 
- `category` (string): The task category under which the question falls into. The task taxonomy could refer to [AlignBench](https://github.com/THUDM/AlignBench).
- `reference` (string): This provides a reference answer to the question. 
- `response_1` (string): A LLM's response to the question. 
- `response_2` (string): Another LLM's response to the question. 

### Referenced Pointwise Grading

To acquire the referenced pointwise grading results of the training data, run this command:

```bash
cd data
python pointwise_reference_based_judgement.py \
    --input_path <path_to_training_data> \
    --output_path pointwise_reference_based.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pointwise_reference_based_judgement_1` and `pointwise_reference_based_judgement_2` fields are the referenced pointwise grading results of `response_1` and `response_2`  respectively.

### Referenced Pointwise Grading to Referenced Pairwise Comparison ($$f_{P2P}$$ in Path#1)

To acquire the referenced pairwise comparison results from referenced pointwise grading results ($$f_{P2P}$$ in Path#1 in our paper), run this command:

```bash
cd data
python P2P_1.py \
    --input_path pointwise_reference_based.jsonl \
    --output_path P2P_1.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pointwise_reference_based_to_pairwise_judgement` field is the referenced pairwise comparison result of `response_1` and `response_2`.

### Referenced Pairwise Comparison to Reference-Free Pairwise Comparison ($$f_{R2RF}$$ in Path#1)

To acquire the reference-free pairwise comparison results from reference-based pairwise comparison results ($$f_{R2RF}$$ in Path#1 in our paper), run this command:

```bash
cd data
python R2RF_1.py \
    --input_path P2P_1.jsonl \
    --output_path R2RF_1.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pairwise_reference_based_to_reference_free_judgement` field is the reference-free pairwise comparison result of `response_1` and `response_2`.

### Referenced Pointwise Grading to Reference-Free Pointwise Grading ($$f_{R2RF}$$ in Path#2)

To acquire the reference-free pointwise grading results from referenced pointwise grading results ($$f_{R2RF}$$ in Path#2 in our paper), run this command:

```bash
cd data
python R2RF_2.py \
    --input_path pointwise_reference_based.jsonl \
    --output_path R2RF_2.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pointwise_reference_free_judgement_1` and `pointwise_reference_free_judgement_2` fields are the reference-free pointwise grading results of `response_1` and `response_2` , respectively.

### Reference-Free Pointwise Grading to Reference-Free Pairwise Comparison ($$f_{P2P}$$ in Path#2)

To acquire the reference-free pairwise comparison results from reference-free pointwise grading results ($$f_{P2P}$$ in Path#2 in our paper), run this command:

```bash
cd data
python P2P_2.py \
    --input_path R2RF_2.jsonl \
    --output_path P2P_2.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pointwise_reference_free_to_pairwise_judgement` field is the reference-free pairwise comparison result of `response_1` and `response_2`.

## Inference

### Pointwise Grading

Some of the test samples of AlignBench are placed in `data/evaluation/pointwise`.  You could preprocess your own judge samples referring to these files. 

The format of judge samples is as follows:

- `id` (integer): A unique identifier for the instance.
- `question` (string): The actual user query. 
- `category` (string): The task category under which the question falls into. You can use your task taxonomy and prepare a `category2type.json` referring to `inference/config/category2type_alignbench.json`
- `reference` (string): This provides a reference answer to the question (if the referenced setting is adopted). 
- `response` (string): A LLM's response to the question. 
- `human_score` (float): The human rating score of the LLM's response. 

Run this command to make pointwise grading judgments via CritiqueLLM:

```bash
cd inference
python inference.py \
    --model_path <path_to_critiquellm> \
    --input_path <path_to_judge_samples> \
    --output_path <path_to_results_file> \
    --output_name <name_of_results_file> \
    --pointwise True \
    --reference_free <whether_to_use_reference_free_setting> \
    --maps <path_to_category2type.json>
```

### Pairwise Comparison

The test samples of AUTO-J (Eval-P), LLMEval, and some of the test samples of AlignBench are placed in `data/evaluation/pairwise`.  You could preprocess your own judge samples referring to these files. 

The format of judge samples is as follows:

- `id` (integer): A unique identifier for the instance.

- `question` (string): The actual user query. 

- `category` (string): The task category under which the question falls into. You can use your task taxonomy and prepare a `category2type.json` referring to `inference/config/category2type_alignbench.json`

- `reference` (string): This provides a reference answer to the question (if the referenced setting is adopted). 

- `response_1` (string): A LLM's response to the question. 

- `response_2` (string): Another LLM's response to the question. 

- `label` (integer): The human preference label of `response_1` and `response_2`. `0` denotes that `response_1` wins, `1` represents that `response_2` wins, and `2` indicates tie.

Run this command to make pairwise comparison judgments via CritiqueLLM:

```bash
cd inference
python inference.py \
    --model_path <path_to_critiquellm> \
    --input_path <path_to_judge_samples> \
    --output_path <path_to_results_file> \
    --output_name <name_of_results_file> \
    --pointwise False \
    --reference_free <whether_to_use_reference_free_setting> \
    --maps <path_to_category2type.json>
```

## Evaluation

### Pointwise Grading

To calculate the correlation between pointwise grading results produced by CritiqueLLM and human ratings, run this command:

```bash
cd evaluation
python eval_pointwise.py --input_path <path_to_pointwise_grading_results> --output <path_to_results_file>
```

Note that `path_to_pointwise_grading_results` refers to the result file of inference.

### Pairwise Comparison

To calculate the agreement and consistency rates of pairwise comparison results produced by CritiqueLLM, run this command:

```bash
cd evaluation
python eval_pairwise.py --input_path <path_to_pairwise_comparison_results> --output <path_to_results_file>
```

Note that `path_to_pairwise_comparison_results` refers to the result file of inference.

### Citation

```
@inproceedings{ke-etal-2024-critiquellm,
    title = "CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation",
    author = "Pei Ke and Bosi Wen and Zhuoer Feng and Xiao Liu and Xuanyu Lei and Jiale Cheng and Shengyuan Wang and Aohan Zeng and Yuxiao Dong and Hongning Wang and Jie Tang and Minlie Huang",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
    year = "2024",
}
```

Please kindly cite our paper if this paper and the codes are helpful.
