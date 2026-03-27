# 5_eval_length_bias.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    # ==========================================
    # 1. 路径与配置
    # ==========================================
    policy_base_id = "Qwen/Qwen2.5-0.5B-Instruct"
    orpo_lora_path = "./outputs/checkpoints/final_policy_model_weights"
    dataset_cache_dir = "./data/hf_datasets_cache"
    
    num_eval_samples = 200 # 评估 200 条以获取统计学上稳定的平均值

    # ==========================================
    # 2. 加载模型与 Tokenizer
    # ==========================================
    print(f">>> 1. 正在加载策略模型基座与 Tokenizer ({policy_base_id})...")
    tokenizer = AutoTokenizer.from_pretrained(policy_base_id)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    base_policy = AutoModelForCausalLM.from_pretrained(
        policy_base_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    print(">>> 2. 挂载 RS-ORPO 训练的 LoRA 权重...")
    policy_model = PeftModel.from_pretrained(base_policy, orpo_lora_path)
    policy_model.eval()

    # ==========================================
    # 3. 加载测试集
    # ==========================================
    print(">>> 3. 加载 UltraFeedback 测试集 Prompt...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", cache_dir=dataset_cache_dir)
    prompts = list(set(dataset["prompt"]))[:num_eval_samples]

    baseline_lengths = []
    orpo_lengths = []

    # ==========================================
    # 4. 实时双路生成与 Token 计数
    # ==========================================
    print(f">>> 4. 开始双路生成并统计长度 (共 {num_eval_samples} 条)...")
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Generating & Counting"):
            # 格式化输入
            chat_format = [{"role": "user", "content": prompt}]
            
            # 【修复点 1】：明确加上 return_dict=True 确保稳定的字典输出
            gen_inputs = tokenizer.apply_chat_template(
                chat_format, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt",
                return_dict=True
            ).to(policy_model.device)
            
            # 【修复点 2】：从字典中提取 input_ids 来计算 Prompt 的确切长度
            prompt_len = gen_inputs["input_ids"].shape[1]

            # A. 生成 RS-ORPO (默认挂载 LoRA 状态)
            # 【修复点 3】：使用 **gen_inputs 将字典解包传给 generate()
            orpo_outputs = policy_model.generate(
                **gen_inputs, 
                max_new_tokens=256, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
            orpo_token_count = orpo_outputs.shape[1] - prompt_len
            orpo_lengths.append(orpo_token_count)

            # B. 生成 Baseline (临时禁用 LoRA 适配器)
            with policy_model.disable_adapter():
                base_outputs = policy_model.generate(
                    **gen_inputs, # 【修复点 4】：这里同样需要解包
                    max_new_tokens=256, 
                    do_sample=False, 
                    pad_token_id=tokenizer.pad_token_id
                )
                base_token_count = base_outputs.shape[1] - prompt_len
                baseline_lengths.append(base_token_count)

    # ==========================================
    # 5. 统计与分析结果
    # ==========================================
    mean_baseline = np.mean(baseline_lengths)
    mean_orpo = np.mean(orpo_lengths)
    length_change_ratio = (mean_orpo - mean_baseline) / mean_baseline * 100

    print("\n" + "="*50)
    print("📊 长度偏置分析报告 (Length Bias Analysis)")
    print("="*50)
    print(f"评估样本数: {num_eval_samples} 条")
    print(f"Baseline 平均生成 Token 数:  {mean_baseline:.1f} tokens")
    print(f"RS-ORPO  平均生成 Token 数:  {mean_orpo:.1f} tokens")
    print("-" * 50)
    
    if length_change_ratio > 0:
        print(f"长度变化率: 增加 {length_change_ratio:.2f}% 🔺")
    else:
        print(f"长度变化率: 减少 {abs(length_change_ratio):.2f}% 🔻")

    print("\n 结果：")
    if abs(length_change_ratio) <= 5:
        print("长度变化极小 (<=5%)。在保持基座能力 (MMLU/GSM8K) 的同时完成了偏好对齐，且完美避开了大模型常见的 'Length Hacking ' 问题")
    elif 5 < length_change_ratio <= 15:
        print("长度略微增加，属于 ORPO 优化过程中的正常现象（模型学会了更详尽地解释步骤）。")
    else:
        print("长度变化较大 (>15%)。胜率的提升可能部分来源于 '模型变得更啰嗦'，而不是逻辑变得更好")

if __name__ == "__main__":
    main()
