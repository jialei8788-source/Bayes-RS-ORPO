# 6_llm_as_a_judge.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

import json
import random
import torch
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. API 裁判配置
# ==========================================
# 请替换为你新生成的 API Key
client = OpenAI(
    api_key="YOUR API KEY", 
    base_url="YOUR URL"
)

# 强烈建议使用通用能力更强的 qwen-plus 或 qwen-max/GPT-4o 作为裁判
JUDGE_MODEL_ID = "qwen-plus" #qwen-max  你也可以改回 "qwen2.5-math-7b-instruct"

JUDGE_SYSTEM_PROMPT = """你是一个公正的大语言模型评判专家。你需要评估两个AI助手对同一个用户指令的回答。
请从逻辑正确性、帮助程度、指令遵循度和安全性四个维度进行评估。
请先给出你的详细分析过程，最后在新的一行严格按照以下格式输出你的最终结论：
如果助手A更好，输出：[[A]]
如果助手B更好，输出：[[B]]
如果两者同样好或同样差，输出：[[Tie]]"""

def evaluate_pair(prompt, ans_model1, ans_model2):
    """调用 API 评判两段回答的胜负"""
    # 随机打乱顺序以消除 API 的位置偏见 (Position Bias)
    is_swapped = random.choice([True, False])
    if is_swapped:
        ans_a, ans_b = ans_model2, ans_model1
    else:
        ans_a, ans_b = ans_model1, ans_model2

    user_content = f"【用户指令】\n{prompt}\n\n【助手A的回答】\n{ans_a}\n\n【助手B的回答】\n{ans_b}"

    response = client.chat.completions.create(
        model=JUDGE_MODEL_ID,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.0, # 评测必须是贪心采样以保证稳定
        max_tokens=1024
    )
    
    judgment = response.choices[0].message.content
    
    # 解析裁判给出的结论
    if "[[A]]" in judgment:
        winner = "Model2" if is_swapped else "Model1"
    elif "[[B]]" in judgment:
        winner = "Model1" if is_swapped else "Model2"
    else:
        winner = "Tie"
        
    return winner, judgment


def main():
    # ==========================================
    # 2. 路径与本地模型配置
    # ==========================================
    policy_base_id = "Qwen/Qwen2.5-0.5B-Instruct"
    orpo_lora_path = "/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/outputs/checkpoints/final_policy_model_weights"
    dataset_cache_dir = "/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/data/hf_datasets_cache"
    
    # 由于 API 调用需要时间和成本，建议评估 100~200 条即可
    num_eval_samples = 100 

    print(f">>> 1. 正在加载本地策略模型基座 ({policy_base_id})...")
    tokenizer = AutoTokenizer.from_pretrained(policy_base_id)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    base_policy = AutoModelForCausalLM.from_pretrained(policy_base_id, device_map="auto", torch_dtype=torch.bfloat16)
    
    print(">>> 2. 挂载 RS-ORPO 训练的 LoRA 权重...")
    policy_model = PeftModel.from_pretrained(base_policy, orpo_lora_path)
    policy_model.eval()

    print(">>> 3. 加载 UltraFeedback 测试集 Prompt...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs", cache_dir=dataset_cache_dir)
    prompts = list(set(dataset["prompt"]))[:num_eval_samples]

    # ==========================================
    # 3. 实时生成 + API 打分
    # ==========================================
    results = {"baseline_wins": 0, "orpo_wins": 0, "ties": 0}
    detailed_logs = [] # 用于保存带有裁判理由的详细报告，面试利器！

    print(f">>> 4. 开始双路生成并调用云端 API 进行裁判 (共 {num_eval_samples} 条)...")
    for prompt in tqdm(prompts, desc="Evaluating"):
        # 准备输入
        chat_format = [{"role": "user", "content": prompt}]
        gen_inputs = tokenizer.apply_chat_template(
            chat_format, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(policy_model.device)
        prompt_len = gen_inputs.shape[1]

        with torch.no_grad():
            # A. 生成 RS-ORPO (默认状态)
            orpo_outputs = policy_model.generate(gen_inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            orpo_ans = tokenizer.decode(orpo_outputs[0][prompt_len:], skip_special_tokens=True)

            # B. 生成 Baseline (禁用 LoRA 适配器)
            with policy_model.disable_adapter():
                base_outputs = policy_model.generate(gen_inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                baseline_ans = tokenizer.decode(base_outputs[0][prompt_len:], skip_special_tokens=True)

        # C. 呼叫 API 进行评判 (加入异常处理防止网络波动中断循环)
        try:
            winner, reason = evaluate_pair(prompt, baseline_ans, orpo_ans)
            
            if winner == "Model1":
                results["baseline_wins"] += 1
            elif winner == "Model2":
                results["orpo_wins"] += 1
            else:
                results["ties"] += 1

            # 记录详细日志
            detailed_logs.append({
                "prompt": prompt,
                "baseline_ans": baseline_ans,
                "orpo_ans": orpo_ans,
                "winner": "Baseline" if winner == "Model1" else ("RS-ORPO" if winner == "Model2" else "Tie"),
                "judge_reason": reason
            })
        except Exception as e:
            print(f"\n[API 错误] 题目评测失败，已跳过: {e}")

    # ==========================================
    # 4. 统计与输出报告
    # ==========================================
    total_valid = sum(results.values())
    if total_valid > 0:
        win_rate = results['orpo_wins'] / total_valid * 100
        print("\n" + "="*50)
        print("🏆 LLM-as-a-Judge 主观胜率报告")
        print("="*50)
        print(f"有效评估题数: {total_valid} / {num_eval_samples}")
        print(f"RS-ORPO 胜出: {results['orpo_wins']} 场 (胜率: {win_rate:.1f}%)")
        print(f"Baseline 胜出: {results['baseline_wins']} 场")
        print(f"平局局数:    {results['ties']} 场")
        print("-" * 50)
        
        # 保存裁判的详细分析
        log_path = "./outputs/results/llm_judge_detailed_report.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(detailed_logs, f, ensure_ascii=False, indent=2)
        print(f"📝 裁判详细分析日志已保存至: {log_path} ")

if __name__ == "__main__":
    main()
