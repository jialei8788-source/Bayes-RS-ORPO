# 1_train_rm.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' #模型下载
os.environ["CUDA_VISIBLE_DEVICES"] = "1" #单卡运行
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model  #,prepare_model_for_kbit_training
from trl import RewardTrainer, RewardConfig



def main():
    dataset_cache_dir = "/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/data/hf_datasets_cache"
    os.makedirs(dataset_cache_dir, exist_ok=True)
    os.makedirs("/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/outputs/checkpoints", exist_ok=True)
    #EssentialAI/rnj-1-instruct
    #Qwen/Qwen2.5-7B-Instruct
    #Qwen/Qwen2.5-0.5B
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    print(">>> 1. 加载 Tokenizer 与基座模型 -4bit量化(分类头)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1, # 标量输出
        device_map="auto",       
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # 👈 开启 Flash Attention
    )
    rm_model.config.pad_token_id = tokenizer.pad_token_id

    # 关键：QLoRA / 4bit 训练前必须先做这个
    #rm_model = prepare_model_for_kbit_training(rm_model)
    
    print(">>> 2. 注入 LoRA (开启 lora_dropout 为后续 MC Dropout 做准备)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1, # 关键：用于不确定性估计
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    rm_model = get_peft_model(rm_model, peft_config)

    print(">>> 3. 加载 UltraFeedback Binarized 偏好数据集...")
    # 加载真实数据集。UltraFeedback 包含超 6 万条高质量成对偏好数据。
    raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs",cache_dir=dataset_cache_dir)
    
    # [可选项] 考虑到完整数据集训练时间较长，初期调试时可以先截取一个子集测试跑通
    #raw_dataset = raw_dataset.select(range(1000))
    print(f"当前训练集样本数: {len(raw_dataset)}")

    def preprocess_function(examples):
        new_examples = {
            "chosen": [], 
            "rejected": []
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            # 关键修改：tokenize=False 让它返回拼接好的纯文本字符串，而不是张量
            formatted_chosen = tokenizer.apply_chat_template(
                chosen, tokenize=False
            )
            formatted_rejected = tokenizer.apply_chat_template(
                rejected, tokenize=False
            )
            
            new_examples["chosen"].append(formatted_chosen)
            new_examples["rejected"].append(formatted_rejected)
            
        return new_examples

    print(">>> 4. 数据预处理与 Tokenization (这可能需要几分钟)...")
    tokenized_dataset = raw_dataset.map(
        preprocess_function, 
        batched=True,
        num_proc=4, 
        remove_columns=raw_dataset.column_names # 删掉原始的复杂列表列，只留下纯文本的 chosen 和 rejected
    )

    print(">>> 5. 启动 RM 训练...")
    reward_config = RewardConfig(
        output_dir="/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/outputs/checkpoints/rm_tmp",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8, # 根据您的显存大小调整
        learning_rate=1e-5,
        optim="paged_adamw_8bit",
        logging_steps=10,
        max_length=1024, # 提升 max_length 以容纳更长的复杂推理数据
        num_train_epochs=1,
        save_strategy="no",
        bf16=True, # 开启 bfloat16 加速
    )

    trainer = RewardTrainer(
        model=rm_model,
        processing_class=tokenizer,
        args=reward_config,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print(">>> 6. 保存 RM 权重...")
    trainer.model.save_pretrained("/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/outputs/checkpoints/rm_model_final")
    tokenizer.save_pretrained("/home/gaostudent/LeiJia/NLP/myproject/proj3_RS_ORPO/outputs/checkpoints/rm_model_final")
    print("RM 训练完成！")

if __name__ == "__main__":
    main()