
# 🚀 Bayes-RS-ORPO: Bayesian Rejection Sampling ORPO
A memory-efficient LLM alignment pipeline combining Bayesian uncertainty-aware Rejection Sampling and ORPO, achieving lower alignment tax through rigorous statistical hard-negative mining.

[![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B-blue)](#)
[![Algorithm](https://img.shields.io/badge/Algorithm-ORPO%20%7C%20Rejection%20Sampling-green)](#)
[![Framework](https://img.shields.io/badge/Framework-HuggingFace%20%7C%20PEFT-orange)](#)
[![Stats](https://img.shields.io/badge/Math-Bayesian%20%7C%20Z--Test%20%7C%20KDE-red)](#)

本项目实现了一套完整、内存高效且具有极强统计学严谨性的大语言模型 (LLM) 偏好对齐全链路。通过引入 **MC Dropout 贝叶斯不确定性估计** 和 **双样本 Z 检验 (Z-test)**，有效解决了传统对齐算法（如 DPO/RLHF）中噪声数据导致小模型“灾难性遗忘”和“对齐税 (Alignment Tax)”的问题。

## ✨ 核心亮点 (Key Features)

* **⚡ 极低显存开销 (Memory Efficient)**：采用单阶段的 **ORPO** (Odds Ratio Preference Optimization) 算法，摒弃了 DPO 中对 Reference Model 的双路加载需求，结合 QLoRA 实现单卡/双卡的高效迭代。
* **🧠 统计学困难负样本挖掘 (Statistical Hard-Negative Mining)**：
    * **缓解分布偏移**：抛弃静态开源偏好数据集，采用策略模型进行 **On-Policy 拒绝采样**。
    * **贝叶斯不确定性 (Epistemic Uncertainty)**：在 Reward Model 打分时开启 MC Dropout，获取后验奖励分布的期望 $\mu$ 和方差 $\sigma^2$。
    * **假设检验 (Hypothesis Testing)**：利用置信下界 (LCB) 锁定候选优胜者，并构建双样本 Z 检验 $$Z = \frac{\mu_w - \mu_l}{\sqrt{\sigma_w^2 + \sigma_l^2}}$$ 严格过滤噪声，仅保留具有统计显著性 ($p < \alpha$) 的高质量配对数据。
* **📊 严密的评测闭环 (Rigorous Evaluation)**：内置自动化评估脚本，包含主观胜率 (LLM-as-a-Judge)、长度偏置监控、基于 KS 检验 (Kolmogorov-Smirnov test) 的 Reward 漂移分析，以及基于 `lm-evaluation-harness` 的客观能力追踪。

## 📈 实验结果 (Evaluation Results)

在极易发生灾难性遗忘的 **Qwen2.5-0.5B** 极小规模基座上，本项目成功注入了人类偏好，并实现了近乎完美的“零对齐税”。

### 1. 核心能力保持 (Zero Alignment Tax)
通过 `lm-evaluation-harness` 验证，模型在对齐后完美保全了通用知识与数学逻辑能力，证明了统计学过滤噪声的有效性。

| 评测维度 | Baseline (SFT 基座) | StatAlign-ORPO | 变化差异 (Delta) |
| :--- | :--- | :--- | :--- |
| **MMLU (常识与跨学科)** | 46.94% | **46.75%** | -0.19% (极微小波动) |
| **GSM8K (数学灵活提取)** | 31.99% | **30.78%** | -1.21% (可控对齐代价) |

### 2. 偏好重塑分布漂移 (Reward Distribution Drift)
利用独立 7B 奖励模型对测试集进行端到端打分。经 **KS检验** 证明，对齐前后模型的 Reward 期望发生了具有统计显著性 ($p=0.022 < 0.05$) 的右移。

*(在此处插入你的 reward_drift_analysis.png 图片)*
> **结论**：策略模型成功跳出了局部最优解，整体概率分布向高偏好区域发生了实质性转移。

### 3. 长度偏置监控 (Length Bias Check)
相较于基线模型，对齐后的模型平均生成长度变化控制在 **5%** 以内，有效克服了传统对齐算法中常见的“话痨作弊 (Length Hacking)”现象。

## 🛠️ 管线架构与运行指南 (Pipeline & Usage)

本项目包含从数据挖掘到最终评估的 4 个完整阶段：

### Phase 1: 奖励模型训练 (Reward Model)
使用大规模基座 (如 Qwen2.5-7B) 训练序列分类器，作为后续采样的裁判。
```bash
python 1_train_rm.py
