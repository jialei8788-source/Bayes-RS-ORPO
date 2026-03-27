# 2_generate_and_mine.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import json
import random
import tqdm
from scipy.stats import norm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

def get_rm_uncertainty_scores(rm_model, tokenizer, prompt, generations, k_samples=5):
    """开启 MC Dropout 获取后验分布的均值与方差"""
    rm_model.train() # 强制开启 Dropout
    means, variances = np.zeros(len(generations)), np.zeros(len(generations))
    
    with torch.no_grad():
        for i, gen in enumerate(generations):
            # 使用与 RM 训练时完全一致的对话模板进行拼接
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": gen}
            ]
            inputs = tokenizer.apply_chat_template(
                conversation, 
                tokenize=True, 
                return_dict=True, 
                return_tensors="pt"
            ).to(rm_model.device)
            
            scores = []
            for _ in range(k_samples):
                score = rm_model(**inputs).logits[0][0].item()
                scores.append(score)
                
            means[i] = np.mean(scores)
            variances[i] = np.var(scores, ddof=1) if k_samples > 1 else 0
            
    rm_model.eval() # 恢复评估模式
    return means, variances

def mine_hard_negatives(generations, means, variances, alpha=0.05, lambda_1=1.0):
    """基于统计显著性的困难负样本挖掘"""
    n_samples = len(generations)
    if n_samples < 2: return None, None
        
    std_devs = np.sqrt(variances)
    lcb_scores = means - lambda_1 * std_devs
    winner_idx = np.argmax(lcb_scores)
    
    mu_w, var_w = means[winner_idx], variances[winner_idx]
    z_alpha = norm.ppf(1 - alpha)
    
    valid_loser_indices = []
    for i in range(n_samples):
        if i == winner_idx: continue
        mu_i, var_i = means[i], variances[i]
        # 构建假设检验：(mu_w - mu_i) / sqrt(var_w + var_i) > Z_alpha
        test_statistic = (mu_w - mu_i) / np.sqrt(var_w + var_i + 1e-8)
        
        if test_statistic > z_alpha:
            valid_loser_indices.append(i)
            
    if not valid_loser_indices:
        return None, None
        
    # 在满足显著性约束的子集中，寻找均值最大的作为困难负样本
    loser_idx = valid_loser_indices[np.argmax(means[valid_loser_indices])]
    return generations[winner_idx], generations[loser_idx]

def main():
    dataset_cache_dir = "/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/data/hf_datasets_cache"
    
    # 【修改点1】：明确分离策略模型和RM模型的ID
    policy_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    rm_base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f">>> 1. 加载 Tokenizer 与策略模型 ({policy_model_id} 用于生成)...")
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_id)
    if policy_tokenizer.pad_token is None: 
        policy_tokenizer.pad_token = policy_tokenizer.eos_token
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    print(f">>> 2. 加载 RM Tokenizer 与 RM 模型 ({rm_base_model_id})...")
    # 【修改点2】：为RM单独加载Tokenizer，防止词表或模板参数冲突
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_base_model_id)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token

    base_rm = AutoModelForSequenceClassification.from_pretrained(
        rm_base_model_id, 
        num_labels=1, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    base_rm.config.pad_token_id = rm_tokenizer.pad_token_id
    
    # 加载带有 LoRA 权重的 RM (此时 7B 的 LoRA 权重将完美适配 7B 的 base_rm)
    rm_model = PeftModel.from_pretrained(base_rm, "/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/outputs/checkpoints/rm_model_final")

    print(f">>> 3. 从 Hugging Face 加载 UltraFeedback Prompt 题库...")
    prompt_dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized", 
        split="train_prefs",
        cache_dir=dataset_cache_dir 
    )
    
    # 提取唯一的 Prompt 并随机采样 
    all_prompts = list(set(prompt_dataset["prompt"]))
    random.seed(42)
    prompts = random.sample(all_prompts, min(2000, len(all_prompts)))
    
    print(f"成功加载并抽取了 {len(prompts)} 条独立 Prompt 开始采样。")
    
    final_dataset = []
    
    print(">>> 4. 开始在线采样与统计筛选 (生成阶段较慢，请耐心等待)...")
    for prompt in tqdm.tqdm(prompts, desc="Sampling & Mining"):
        # 构造用于生成的 Chat Template 前缀 (使用 policy_tokenizer)
        chat_formatted_prompt = [{"role": "user", "content": prompt}]
        inputs = policy_tokenizer.apply_chat_template(
            chat_formatted_prompt, 
            tokenize=True, 
            add_generation_prompt=True, # 引导模型输出 Assistant 回答
            return_tensors="pt"
        ).to(policy_model.device)
        
        prompt_length = inputs.shape[1]
        
        # 策略模型生成 N=5 个候选回答
        outputs = policy_model.generate(
            inputs, 
            max_new_tokens=256, # 限制单次生成长度防 OOM
            num_return_sequences=5, 
            temperature=0.8, 
            do_sample=True,
            pad_token_id=policy_tokenizer.pad_token_id
        )
        
        # 仅截取新生成的 Token 部分 (使用 policy_tokenizer)
        generations = [
            policy_tokenizer.decode(out[prompt_length:], skip_special_tokens=True) 
            for out in outputs
        ]
        
        # 评分与筛选 (【修改点3】：这里必须传入 rm_tokenizer 供 RM 模型计算使用)
        means, variances = get_rm_uncertainty_scores(rm_model, rm_tokenizer, prompt, generations)
        winner, loser = mine_hard_negatives(generations, means, variances, alpha=0.05)
        
        if winner and loser:
            final_dataset.append({
                "prompt": prompt, 
                "chosen": winner, 
                "rejected": loser
            })

    print(f"\n>>> 5. 筛选完毕！从 {len(prompts)} 个 Prompt 中保留了 {len(final_dataset)} 条有效偏好对。")
    print("正在保存至 /home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/data/rs_mined_dataset.json ...")
    with open("/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/data/rs_mined_dataset.json", "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    print("拒绝采样阶段完成！可以准备进行最终的 ORPO 微调了。")

if __name__ == "__main__":
    main()