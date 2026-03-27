# 4_eval_reward_drift.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 让脚本同时使用两张卡，自动分配显存
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ks_2samp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

def main():
    # ==========================================
    # 1. 路径与配置 (请确保与你的实际路径一致)
    # ==========================================
    policy_base_id = "Qwen/Qwen2.5-0.5B-Instruct"
    orpo_lora_path = "./outputs/checkpoints/final_policy_model_weights"
    
    rm_base_id = "Qwen/Qwen2.5-7B-Instruct"
    rm_lora_path = "./outputs/checkpoints/rm_model_final"
    
    dataset_cache_dir = "./data/hf_datasets_cache"
    num_eval_samples = 200 # 评估 200 条数据画分布图足够了

    # ==========================================
    # 2. 加载模型 (极客技巧：全量加载)
    # ==========================================
    print(">>> 1. 正在加载生成模型 (Baseline + ORPO_LoRA)...")
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_base_id)
    if policy_tokenizer.pad_token is None: policy_tokenizer.pad_token = policy_tokenizer.eos_token
    
    # 加载 0.5B 基座，并贴上 ORPO 训练出来的 LoRA
    base_policy = AutoModelForCausalLM.from_pretrained(policy_base_id, device_map="auto", dtype=torch.bfloat16)
    policy_model = PeftModel.from_pretrained(base_policy, orpo_lora_path)
    policy_model.eval()

    print(">>> 2. 正在加载奖励模型 (RM_Base + RM_LoRA)...")
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_base_id)
    if rm_tokenizer.pad_token is None: rm_tokenizer.pad_token = rm_tokenizer.eos_token
    
    base_rm = AutoModelForSequenceClassification.from_pretrained(rm_base_id, num_labels=1, device_map="auto", dtype=torch.bfloat16)
    base_rm.config.pad_token_id = rm_tokenizer.pad_token_id
    rm_model = PeftModel.from_pretrained(base_rm, rm_lora_path)
    rm_model.eval()

    # ==========================================
    # 3. 准备评测数据
    # ==========================================
    print(">>> 3. 加载 UltraFeedback 测试集...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", cache_dir=dataset_cache_dir)
    # 抽取没见过的 Prompt
    prompts = list(set(dataset["prompt"]))[:num_eval_samples]

    baseline_scores = []
    orpo_scores = []

    # ==========================================
    # 4. 生成与打分流水线
    # ==========================================
    print(f">>> 4. 开始双路生成与 RM 打分 (共 {num_eval_samples} 条)...")
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Evaluating"):
            # A. 准备生成 Prompt
            chat_format = [{"role": "user", "content": prompt}]
            gen_inputs = policy_tokenizer.apply_chat_template(chat_format, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(policy_model.device)
            prompt_len = gen_inputs.shape[1]

            # B. 生成 RS-ORPO 的回答 (默认状态，LoRA开启)
            orpo_outputs = policy_model.generate(gen_inputs, max_new_tokens=256, do_sample=False, pad_token_id=policy_tokenizer.pad_token_id)
            orpo_ans = policy_tokenizer.decode(orpo_outputs[0][prompt_len:], skip_special_tokens=True)

            # C. 生成 Baseline 的回答 (极客技巧：临时禁用 LoRA 适配器，瞬间退化为基线模型)
            with policy_model.disable_adapter():
                base_outputs = policy_model.generate(gen_inputs, max_new_tokens=256, do_sample=False, pad_token_id=policy_tokenizer.pad_token_id)
                baseline_ans = policy_tokenizer.decode(base_outputs[0][prompt_len:], skip_special_tokens=True)

            # D. RM 打分
            # 拼装给 RM 看的对话
            conv_base = [{"role": "user", "content": prompt}, {"role": "assistant", "content": baseline_ans}]
            conv_orpo = [{"role": "user", "content": prompt}, {"role": "assistant", "content": orpo_ans}]
            
            rm_input_base = rm_tokenizer.apply_chat_template(conv_base, tokenize=True, return_dict=True, return_tensors="pt").to(rm_model.device)
            rm_input_orpo = rm_tokenizer.apply_chat_template(conv_orpo, tokenize=True, return_dict=True, return_tensors="pt").to(rm_model.device)
            
            # 获取标量得分
            score_base = rm_model(**rm_input_base).logits[0][0].item()
            score_orpo = rm_model(**rm_input_orpo).logits[0][0].item()

            baseline_scores.append(score_base)
            orpo_scores.append(score_orpo)

    # ==========================================
    # 5. 统计检验与绘图
    # ==========================================
    print("\n>>> 5. 计算统计学指标与绘制分布图...")
    
    mean_base = np.mean(baseline_scores)
    mean_orpo = np.mean(orpo_scores)
    
    # KS 检验 (证明分布漂移在统计学上是显著的)
    ks_stat, p_value = ks_2samp(baseline_scores, orpo_scores)
    
    print("-" * 40)
    print(f"Baseline 平均 Reward: {mean_base:.4f}")
    print(f"RS-ORPO  平均 Reward: {mean_orpo:.4f}")
    print(f"Reward 提升幅度: {(mean_orpo - mean_base):.4f}")
    print(f"KS-Test p-value: {p_value:.2e} (如果 < 0.05 则分布显著改变)")
    print("-" * 40)

    # 绘制更美观的 KDE 分布图
    plt.figure(figsize=(10, 6), dpi=150)
    sns.kdeplot(baseline_scores, fill=True, label=f'Baseline (SFT) Mean: {mean_base:.2f}', color='#1f77b4', alpha=0.5, linewidth=2)
    sns.kdeplot(orpo_scores, fill=True, label=f'RS-ORPO Mean: {mean_orpo:.2f}', color='#2ca02c', alpha=0.5, linewidth=2)
    
    plt.axvline(mean_base, color='#1f77b4', linestyle='dashed', linewidth=2)
    plt.axvline(mean_orpo, color='#2ca02c', linestyle='dashed', linewidth=2)
    
    plt.title("Reward Distribution Shift After RS-ORPO Alignment", fontsize=14, pad=15)
    plt.xlabel("Reward Score (Logits)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    # 在图中加入 P-value 文本框
    plt.text(0.05, 0.95, f'KS-Test $p$-value: {p_value:.2e}', 
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = "./outputs/results/reward_drift_analysis.png"
    plt.savefig(save_path)
    print(f"分布漂移图已保存为: {save_path}")

if __name__ == "__main__":
    main()
