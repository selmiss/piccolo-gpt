<small>[EN](README.md) | [简体中文](README_zh.md) </small>

# Piccolo-GPT

**Paper:** [Piccolo2: General Text Embeddings with Multi-Task Hybrid loss Training](https://arxiv.org/abs/2405.06932)

🚀 **New SOTA on CMTEB** 

🔥 Our general sentence embedding [selmisskilig/piccolo-gpt-zh](https://huggingface.co/selmisskilig/piccolo-gpt-zh) achieves SOTA on the CMTEB Leaderboard with an average score of 70.95. [2024/4/23] One of the Embedding models based on large model training has achieved an average score of 68.03. The repository is the training pipeline code for Piccolo-GPT, a branch of Piccolo Embedding.


## 💡Model Highlights
The Piccolo-GPT-trained piccolo-gpt-zh outperforms most of the models in the combined evaluation of the six tasks on the CMTEB list, and is currently in the top 10. piccolo-gpt-zh primarily utilises efficient multi-task hybrid loss training methods to effectively exploit text data and labels from different downstream tasks. 

 For huggingface model, please refer to our space: [selmisskilig/piccolo-gpt-zh](https://huggingface.co/selmisskilig/piccolo-gpt-zh)
 For training details, please refer to our tech report: https://arxiv.org/abs/2405.06932

## 📖 Repo Details
 In this repo, we release our training code and implement some tricks that are helpful for embedding training, including:
- Multi-task Hybrid Loss Training
- Matryoshka Representation Learning
- Embdding Dimension Scaling
- Task-Homogenous Dataset
- Position Embedding Hierarchical Decomposition 

 In order to save memory, we default use deepspeed-zero1, gradient checkpointing and mix-precision for training. We also provide slurm scripts to help conduct distributed training.

## 🔨 Best Practice
### 1. Environment
```shell
pip install -r requirements.txt
```

### 2. Prepare Dataset
We divide the dataset into three major categories: retrieval/reranking, clustering/classification, abd sts/pair classification, and we adopt different loss functions for different categories. In `data_example` dir, we provide example for these three types of data.

1) `Retri`: For Retrieval and Reranking dataset, we implement standard InfoNCE with in-batch-negative, this dataset contain 4 columns: `text`, `text_pos`, `text_neg`, `type`. Here 'type' is 'retri_contrast'

2) `STS`: For STS and Pair-Classification dataset, we implement the cosent loss, this dataset contain 4 columns: `text`, `text_pair`, `label`, `type`. Here 'type' is 'cosent'
   
3) `Cls`: For Classification and Clustering dataset, we implement the InfoNCE w/o in-batch-negative, this dataset contain 4 columns: `text`, `text_pos`, `text_neg`, `type`. Here 'type' is 'cls_contrast'

The 'type' column indicates the type of the current dataset. We obtain the type of the current batch to apply different loss functions.

### 3. Training
We offer the training script located at `scripts/ft.sh`. Below we've provided explanations for the variables used within the script.

**Environment Variables**  
- ROOT: The absolute path of the repo in your machine. 
- GPUS_PER_NODE: Number of GPUs on a single machine
The default proviede scripts is for single-machine training. If you are in a multi-node distributed training scenario, you should additionally specify the env parameters below:
- WORLD_SIZE: Number of nodes
- RANK: rank of current node, it is often obtained thourgh the SLURM environment variable 
- MASTER_ADDR:MASTER_PORT: Communication port

**Training Variables** 
- MODEL_NAME_OR_PATH: the absolute path of the pretrain model
- DS_PATH: the deepspeed config, we provide a default config in `./data_example/deepspeed_config.json`
- META_PATHS: the list of the dataset. We provide a sample in `meta_lists/piccolo.txt`. Each row consists of two columns: the relative path of the dataset and the number of repeats.
- ROOT_DIRS: The absolute path of the dataset dir

**Run**
```shell
bash scripts/ft.sh
```

## 🤗 **Model List**
| Model|Language||Description|prompt|
|:-|:-:|:-:|:--------------------------------------------:|:---------:|
| [selmisskilig/piccolo-gpt-zh](https://huggingface.co/selmisskilig/piccolo-gpt-zh)                  |    Chinese     |   | finetuning with multi-task hybrid loss training | None |

## Citation
If you find our tech report, models or code helpful, please cite our report or give a star on github or huggingface!  
```bibtex
@misc{2405.06932,
Author = {Junqin Huang and Zhongjie Hu and Zihao Jing and Mengya Gao and Yichao Wu},
Title = {Piccolo2: General Text Embedding with Multi-task Hybrid Loss Training},
Year = {2024},
Eprint = {arXiv:2405.06932},
}
```
