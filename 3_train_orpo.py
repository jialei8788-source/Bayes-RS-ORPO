# 3_train_orpo.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 3_train_orpo.py
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import ORPOConfig, ORPOTrainer


def main():
    #os.makedirs("./checkpoints", exist_ok=True)
    #model_id = "meta-llama/Llama-3-8b-Instruct"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(">>> 1. 加载 Tokenizer 与策略模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print(">>> 2. 注入 LoRA (策略微调)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    policy_model = get_peft_model(policy_model, peft_config)

    print(">>> 3. 加载第二步挖掘的高质量偏好数据集...")
    with open("./data/rs_mined_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 转换为 Hugging Face Dataset
    raw_dataset = Dataset.from_list(data)
    
    # 为了防止 ORPOTrainer 对纯文本直接拼接造成 Llama-3 模板丢失，
    # 我们最好显式地将 prompt, chosen, rejected 转换为对话格式，然后再应用 chat_template
    def format_for_orpo(example):
        # 构建完整的 prompt 对话历史
        prompt_message = [{"role": "user", "content": example["prompt"]}]
        # 严格应用 chat_template 转换为模型认识的特殊 token 字符串
        prompt_str = tokenizer.apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
        
        # 此时的 chosen 和 rejected 只需要直接拼接在 prompt_str 后面即可
        # 因为在第2步中，我们截取的就是模型实际输出的纯文本
        return {
            "prompt": prompt_str,
            "chosen": example["chosen"] + tokenizer.eos_token,   # 加上结束符
            "rejected": example["rejected"] + tokenizer.eos_token # 加上结束符
        }
        
    train_dataset = raw_dataset.map(format_for_orpo)

    print(">>> 4. 启动 ORPO 训练...")
    orpo_config = ORPOConfig(
        output_dir="./outputs/checkpoints/final_policy_model",
        learning_rate=1e-5, #5e-6
        beta=0.4, # ORPO 惩罚权重 (对应几率比损失的 lambda)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_length=1024,
        max_prompt_length=512,
        optim="paged_adamw_8bit",
        logging_steps=5,
        num_train_epochs=5,
        save_strategy="epoch",
        bf16=True, # 开启 bfloat16 加速
        remove_unused_columns=False # 加上这个防止 Dataset 预处理时丢列
    )

    trainer = ORPOTrainer(
        model=policy_model,
        args=orpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer, # <--- 关键修改点：适应最新版 trl
    )

    trainer.train()
    print(">>> 5. 保存最终对齐的模型权重...")
    trainer.model.save_pretrained("./outputs/checkpoints/final_policy_model_weights")
    tokenizer.save_pretrained("./outputs/checkpoints/final_policy_model_weights")
    print("全流程训练完毕")

if __name__ == "__main__":
    main()
