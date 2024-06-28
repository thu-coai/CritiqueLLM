# CRITIQUELLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation

This repository is the official implementation of [CRITIQUELLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation](https://arxiv.org/abs/2311.18702). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Download

You can download CritiqueLLM-6B [here](https://huggingface.co/thu-coai/CritiqueLLM-6B).

## Data Collection

Some Samples of training data are placed in `data/train`.  The format of training data should be as follows.

- `id` (integer): A unique identifier for the instance.
- `question` (string): The actual user query. 
- `category` (string): The task category under which the question falls. The taxonomy of the task could refer to [AlignBench](https://arxiv.org/abs/2311.18743).
- `reference` (string): This provides a reference or standard answer to the question. 
- `response_1` (string): A LLM response to the question. 
- `response_2` (string): Another LLM response to the question. 

### Reference-Based Pointwise Grading

To get the reference-based pointwise grading results of the training data, run this command:

```bash
cd data
python pointwise_reference_based_judgement.py \
    --input_path <path_to_training_data> \
    --output_path pointwise_reference_based.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pointwise_reference_based_judgement_1` and `pointwise_reference_based_judgement_2` fields are the reference-based pointwise grading results of `response_1` and `response_2`  respectively.

### Reference-based Pointwise Grading to Reference-based Pairwise Comparison ($$f_{P2P}$$ in Path#1)

To get the reference-based pairwise comparison results from reference-based pointwise grading results ($$f_{P2P}$$ in Path#1 in our paper), run this command:

```bash
cd data
python P2P_1.py \
    --input_path pointwise_reference_based.jsonl \
    --output_path P2P_1.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pointwise_reference_based_to_pairwise_judgement` field is the reference-based pairwise comparison result of `response_1` and `response_2`.

### Reference-Based Pairwise Comparison to Reference-Free Pairwise Comparison ($$f_{R2RF}$$ in Path#1)

To get the reference-free pairwise comparison results from reference-based pairwise comparison results ($$f_{R2RF}$$ in Path#1 in our paper), run this command:

```bash
cd data
python R2RF_1.py \
    --input_path P2P_1.jsonl \
    --output_path R2RF_1.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pairwise_reference_based_to_reference_free_judgement` field is the reference-free pairwise comparison result of `response_1` and `response_2`.

### Reference-Based Pointwise Grading to Reference-Free Pointwise Grading ($$f_{R2RF}$$ in Path#2)

To get the reference-free pointwise grading results from reference-based pointwise grading results ($$f_{R2RF}$$ in Path#2 in our paper), run this command:

```bash
cd data
python R2RF_2.py \
    --input_path pointwise_reference_based.jsonl \
    --output_path R2RF_2.jsonl \
    --api_key <openai_api_key> \
    --api_base <openai_api_base> \
```

The `pointwise_reference_free_judgement_1` and `pointwise_reference_free_judgement_2` fields are the reference-free pointwise grading results of `response_1` and `response_2`  respectively.

### Reference-Free Pointwise Grading to Reference-Free Pairwise Comparison ($$f_{P2P}$$ in Path#2)

To get the reference-free pairwise comparison results from reference-free pointwise grading results ($$f_{P2P}$$ in Path#2 in our paper), run this command:

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

Samples of AlignBench are placed in `data/evaluation/pointwise`.  You could preprocess judge samples referring to these files. 

The format of judge samples should be as follows:

- `id` (integer): A unique identifier for the instance.
- `question` (string): The actual user query. 
- `category` (string): The task category under which the question falls. You can use your task taxonomy and prepare a `category2type.json` referring to `inference/config/category2type_alignbench.json`
- `reference` (string): This provides a reference or standard answer to the question (If the reference-based setting is used). 
- `response_1` (string): A LLM response to the question. 
- `response_2` (string): Another LLM response to the question. 

run this command to make pairwise  by CritiqueLLM:

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

Samples of AutoJ, LLMEval, and some samples of AlignBench are placed in `data/evaluation/pairwise`.  You could preprocess judge samples referring to these files. 

The format of judge samples should be as follows:

- `id` (integer): A unique identifier for the instance.
- `question` (string): The actual user query. 
- `category` (string): The task category under which the question falls. You can use your task taxonomy and prepare a `category2type.json` referring to `inference/config/category2type_alignbench.json`
- `reference` (string): This provides a reference or standard answer to the question (If the reference-based setting is used). 
- `response` (string): A LLM response to the question. 

run this command to make pointwise comparison judgments by CritiqueLLM:

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

To calculate the correlation between pointwise grading results produced by CritiqueLLM and human ratings, run:

```bash
cd evaluation
python eval_pointwise.py --input_path <path_to_pointwise_grading_results> --output <path_to_results_file>
```

### Pairwise Comparison

To consistency the agreement of pairwise comparison results produced by CritiqueLLM, run:

```bash
cd evaluation
python eval_pairwise.py --input_path <path_to_pairwise_comparison_results> --output <path_to_results_file>
```
