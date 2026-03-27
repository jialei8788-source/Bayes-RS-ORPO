# 1_train_rm.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import RewardTrainer, RewardConfig


def main():
    # -------- 分布式设备绑定：一定放在 main 里 --------
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    current_device = torch.cuda.current_device()

    print(f"[Rank {local_rank}] current_device = cuda:{current_device}")

    dataset_cache_dir = "./data/hf_datasets_cache"
    output_dir = "./outputs/checkpoints"
    os.makedirs(dataset_cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # EssentialAI/rnj-1-instruct
    # Qwen/Qwen2.5-7B-Instruct
    # Qwen/Qwen2.5-0.5B
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    print(">>> 1. 加载 Tokenizer 与基座模型（4bit量化 + 分类头）...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rm_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        quantization_config=quantization_config,
        device_map={"": torch.cuda.current_device()},   # 关键修复
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    rm_model.config.pad_token_id = tokenizer.pad_token_id
    rm_model.config.use_cache = False

    print(">>> 2. 准备 k-bit 训练并注入 LoRA ...")

    # 关键：QLoRA / 4bit 训练前必须先做这个
    rm_model = prepare_model_for_kbit_training(rm_model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    rm_model = get_peft_model(rm_model, peft_config)

    # 可选：打印可训练参数
    rm_model.print_trainable_parameters()

    print(">>> 3. 加载 UltraFeedback Binarized 偏好数据集...")
    raw_dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        cache_dir=dataset_cache_dir,
    )

    # 初期调试先取子集
    raw_dataset = raw_dataset.select(range(1000))
    print(f"当前训练集样本数: {len(raw_dataset)}")

    def preprocess_function(examples):
        new_examples = {
            "chosen": [],
            "rejected": []
        }

        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            formatted_chosen = tokenizer.apply_chat_template(
                chosen,
                tokenize=False,
            )
            formatted_rejected = tokenizer.apply_chat_template(
                rejected,
                tokenize=False,
            )

            new_examples["chosen"].append(formatted_chosen)
            new_examples["rejected"].append(formatted_rejected)

        return new_examples

    print(">>> 4. 数据预处理...")
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=raw_dataset.column_names,
    )

    print(">>> 5. 启动 RewardTrainer 训练...")

    reward_config = RewardConfig(
        output_dir=f"{output_dir}/rm_tmp",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        optim="paged_adamw_8bit",
        logging_steps=10,
        max_length=1024,
        num_train_epochs=1,
        save_strategy="no",
        bf16=True,
        report_to="none",
        ddp_find_unused_parameters=False,   # LoRA + DDP 常见建议
    )

    trainer = RewardTrainer(
        model=rm_model,
        processing_class=tokenizer,
        args=reward_config,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    print(">>> 6. 保存 RM 权重...")
    save_path = f"{output_dir}/rm_model_final"
    trainer.model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("RM 训练完成！")


if __name__ == "__main__":
    main()
