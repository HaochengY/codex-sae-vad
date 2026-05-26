# Frame-Level VAD Compass 训练方案

## 1. 目标

本项目目标是在 3 秒视频 clip 上训练一个基于 SegCompass 思路改造的 frame-level VAD 模型。

输入是一个短视频 clip 和异常规则列表，输出包括：

- 模型生成的推理文本 `<think>...</think>`
- 4 个规则聚焦 token：`<RULE> <RULE> <RULE> <RULE>`
- 每个 slot 的帧级异常预测：clip 内每一帧的异常 `0/1` 序列预测，共 4 个 slot 预测
- 每个 slot 的 confidence

训练数据规模：

- 12144 个 clip
- 每个 clip 约 3 秒
- 每个 clip 有 frame-level label
- 每帧标签为 `0/1`，表示该帧正常或异常

## 2. 整体思路

模型结构复刻 SegCompass 的核心路径：

```text
视频 + 异常规则 prompt
→ MLLM 生成 CoT + 4 个 <RULE> token
→ 读取 <RULE> token hidden states 得到 e_k
→ SAE 编码完整输出 hidden states 得到 sparse activations
→ codebook 特征和 4 个可学习 slot 参数 concat 后送入 Transformer
→ 只取 4 个 slot 参数位置的输出作为 r_k
→ concat(e_k, r_k) + MLP 得到 Q_k
→ Q_k 与视频 ViT features K 做 attention
→ temporal heatmap H_k[t]
→ detection head 输出每个 slot 的帧级异常 0/1 序列预测
→ confidence head 输出每个 slot 的可信度 c_k
```

其中：

```text
K = 4
```

表示固定使用 4 个 anomaly/rule slots。它不是表示一定有 4 个异常，而是最多给模型 4 个候选解释/检测槽位。

## 3. Prompt 设计

训练 prompt 需要强制模型输出推理和 4 个 `<RULE>` token。

推荐模板：

```text
<video>
You are a video anomaly detection assistant.

Known anomaly rule list:
1. Skateboarding is anomalous.
2. Riding a bicycle is anomalous.
3. Running is anomalous.
4. Fighting is anomalous.
5. Driving or riding a vehicle is anomalous.
6. Falling is anomalous.
7. Jumping is anomalous.

Determine whether this video clip contains any behavior that violates the rules above.

Requirements:
1. First, briefly explain the reasoning inside <think> </think>.
2. Then output `Violated rules:` followed by exactly 4 rule-focus tokens. Each token must be written as <RULE>.
3. Do not output Answer or any other extra text.

Format:
<think> your reasoning here </think>
Violated rules:<RULE> <RULE> <RULE> <RULE>
```

训练和评估时都必须保持同一格式。

## 4. Special Token

需要提前注册 special token：

```text
<RULE>
```

要求：

- tokenizer 中必须包含 `<RULE>`
- MLLM embedding 需要 resize
- checkpoint 必须保存 `<RULE>` embedding
- eval 时必须恢复训练时的 `<RULE>` embedding

不要在 eval 时临时随机初始化 `<RULE>`，否则 `e_k` 会不稳定。

## 5. 模型结构

### 5.1 MLLM 输出

MLLM 输入：

```text
video clip + rule prompt
```

MLLM 输出：

```text
<think>...</think>
Violated rules:<RULE> <RULE> <RULE> <RULE>
```

从最后 4 个 `<RULE>` token 的 hidden states 中读取：

```text
e_k ∈ R^{D_llm}, k=1..4
```

### 5.2 SAE 分支

取 MLLM 指定层 hidden states：

```text
z = hidden_states[layer]
```

经过 SAE：

```text
h(z) = SAE(z)
```

得到 sparse activations。

### 5.3 Codebook + Transformer

SAE sparse activations 经过 codebook：

```text
h(z) → codebook → dense concept tokens
```

准备 4 个可学习 slot 参数：

```text
s_k ∈ R^{D_c}, k=1..4
```

将 codebook 输出特征和这 4 个可学习参数 concat 后一起输入 Transformer：

```text
concat(concept_tokens, s_1, s_2, s_3, s_4) → Transformer
```

Transformer 输出后只取最后 4 个 slot 参数位置，作为规则/异常 slot 表征：

```text
r_k ∈ R^{D_c}, k=1..4
```

### 5.4 融合 e_k 和 r_k

每个 slot 融合：

```text
q_k = MLP(concat(e_k, r_k))
```

得到最终 query：

```text
Q = {q_1, q_2, q_3, q_4}
```

### 5.5 视频 ViT 特征作为 K

视频输入经过 ViT/video encoder 得到时序特征：

```text
K_video ∈ R^{T × D_v}
```

其中：

- `T` 是 clip 内采样帧数
- 每个时间位置对应一帧或一个 temporal token

### 5.6 Temporal Attention Heatmap

用 `Q_k` 和视频特征 `K_video` 做 attention：

```text
H_k[t] = attention(q_k, K_video[t])
```

得到：

```text
H ∈ R^{K × T}
```

含义：

- `H_k[t]` 表示第 k 个 slot 在第 t 帧的异常响应强度

### 5.7 Detection Head

检测头基于 `H_k[t]` 输出每个 slot 的帧级异常 logit：

```text
det_logits_k[t]
```

形状：

```text
[K, T]
```

每个 slot 的帧级异常分数为：

```text
slot_frame_score_k[t] = sigmoid(det_logits_k[t])
```

每个 slot 对应一个 clip 内每一帧的异常 `0/1` 序列预测：

```text
slot_frame_pred_k[t] = 1 if slot_frame_score_k[t] >= threshold else 0
```

一共有 4 个 slot 预测。有些 slot 的帧级预测可能不准，使用时必须结合该 slot 的 confidence：

```text
weighted_slot_score_k[t] = sigmoid(conf_logits[k]) * slot_frame_score_k[t]
```

最终帧级分数推荐按 confidence 加权后聚合：

```text
frame_score[t] = max_k weighted_slot_score_k[t]
```

训练时也可以使用 matched slots 单独监督。

### 5.8 Confidence Head

confidence head 输出：

```text
conf_logits ∈ R^K
```

含义：

- `conf_logits[k]` 表示第 k 个 slot 是否捕捉到了一个真实异常片段/规则违反事件

## 6. Frame-Level Label

每个 clip 有帧级标签：

```text
y[t] ∈ {0, 1}
```

其中：

- `0` 表示第 t 帧正常
- `1` 表示第 t 帧异常

3 秒 clip 不一定把所有帧都送入模型，需要增加一个采样参数控制每个 clip 抽多少帧参与训练：

```text
num_train_frames_per_clip = 12
```

默认值为 `12`。如果原始帧数和模型采样帧数不同，需要把 label 对齐到模型采样的 `T` 个时间点。

推荐做法：

```text
sampled_frame_indices = 视频采样帧下标
y_sampled[t] = 原始 label[sampled_frame_indices[t]]
```

## 7. Slot Matching 和 Confidence Target

有 frame-level label 后，可以给 4 个 slot 构造 confidence 监督。

### 7.1 将连续异常帧转为异常片段

从 `y[t]` 中提取连续异常区间：

```text
G_j = [start_j, end_j]
```

例如：

```text
y = 0001111000110
```

得到两个异常片段：

```text
G_1 = frames 3-6
G_2 = frames 10-11
```

### 7.2 计算 slot 与 GT segment 的匹配分数

每个 slot 有 temporal heatmap：

```text
H_k[t]
```

将其 sigmoid 后得到：

```text
P_k[t] = sigmoid(H_k[t])
```

和每个 GT segment mask `G_j[t]` 计算 soft temporal IoU：

```text
IoU(k, j) = sum(P_k * G_j) / sum(P_k + G_j - P_k * G_j)
```

### 7.3 Hungarian Matching

用 Hungarian matching 匹配：

```text
slot k ↔ GT segment j
```

匹配上的 slot：

```text
conf_target[k] = 1
```

没匹配上的 slot：

```text
conf_target[k] = 0
```

如果该 clip 没有异常帧：

```text
conf_target = [0, 0, 0, 0]
```

## 8. Loss 设计

总 loss：

```text
L_total = L_GRPO + λ_frame * L_frame + λ_conf * L_conf + λ_sae * L_sae
```

### 8.1 Format Reward / GRPO

GRPO reward：

```text
R = α * format_score + β * frame_score
```

其中：

```text
format_score
```

检查输出格式是否满足：

- 恰好一个 `<think>...</think>`
- `<think>` 内容非空
- `</think>` 后必须出现 `Violated rules:`
- `Violated rules:` 后能解析出 4 个 `<RULE>` token
- 恰好 4 个 `<RULE>`
- `<RULE>` 出现在 `Violated rules:` 后面
- 不允许输出 `Answer`

格式分要求：

```text
format_score 基础分 = 1.0
```

- 违反任意硬性格式规则，`format_score = 0`
- 格式完全正确，`format_score = 1.0`
- 格式正确但 `<think>` 内容超过 2048 字符，`format_score = 0.9`
- 核心格式正确但 `<think>` 前存在任何非空白文本，或最后一个 special token `<RULE>` 后存在任何非空白文本，`format_score = 0.9`

```text
frame_score
```

用模型生成的 slot/frame prediction 和 frame-level label 计算，可使用：

```text
frame_score = 1 - frame_BCE_normalized
```

或：

```text
frame_score = temporal IoU / Dice
```

GRPO 用于训练：

```text
模型生成 <think> + Violated rules: + 4 个 <RULE> 的策略
```

### 8.2 Frame-Level Detection Loss

帧级检测 loss：

```text
slot_frame_score_k[t] = sigmoid(det_logits_k[t])
weighted_slot_score_k[t] = sigmoid(conf_logits[k]) * slot_frame_score_k[t]
frame_score[t] = max_k weighted_slot_score_k[t]
```

然后：

```text
L_frame_bce = BCE(frame_score, y)
```

可加 Dice：

```text
L_frame_dice = 1 - Dice(frame_score, y)
```

最终：

```text
L_frame = L_frame_bce + λ_dice * L_frame_dice
```

### 8.3 Matched Slot Frame Loss

如果使用 Hungarian matching，则对匹配上的 slot 单独监督：

```text
L_slot_frame = BCEWithLogits(det_logits_k, G_j)
```

其中：

```text
k ↔ j
```

是 Hungarian matching 结果。

未匹配 slot 不参与 frame loss，或只用 confidence loss 压低。

### 8.4 Confidence Loss

confidence loss：

```text
L_conf = BCEWithLogits(conf_logits, conf_target)
```

其中：

```text
conf_target[k] = 1 if slot k matched a GT abnormal segment else 0
```

这对应 SegCompass 中：

```text
matched mask slot → confidence target 1
unmatched mask slot → confidence target 0
```

### 8.5 SAE Loss

如果 SAE 参与训练，可加重构 loss：

```text
L_sae = MSE(z_hat, z)
```

如果 SAE 是预训练并冻结，则可以不加或只记录。

## 9. 训练流程

每个 batch：

1. 读取视频 clip 和 frame-level label
2. 从 3 秒 clip 中按 `num_train_frames_per_clip` 采样训练帧，默认采样 12 帧
3. 将原始 frame-level label 对齐到采样帧
4. 根据异常规则列表构造 prompt
5. MLLM 生成 response：

```text
<think>...</think>
Violated rules:<RULE> <RULE> <RULE> <RULE>
```

6. 读取 `<RULE>` hidden states 得到 `e_k`
7. 读取指定层 hidden states `z`
8. `z → SAE → codebook → concept_tokens`
9. `concat(concept_tokens, 4 个 learnable slot 参数) → Transformer`
10. 只取 4 个 learnable slot 参数位置的输出作为 `r_k`
11. `concat(e_k, r_k) → MLP → q_k`
12. `q_k` attend 视频 ViT features 得到 `H_k[t]`
13. detection head 得到 `det_logits_k[t]`
14. confidence head 得到 `conf_logits[k]`
15. 根据 frame-level label 构造 GT segments
16. Hungarian matching 得到 slot target
17. 计算：

```text
L_GRPO
L_frame
L_conf
L_sae
```

18. 反向传播更新：

```text
MLLM / LoRA
SAE 或 SAE adapter
codebook
Transformer slots
detection head
confidence head
```

训练过程建议实时监控 loss，至少每隔固定 step 输出并写入 `metrics.jsonl`：

```text
step
lr
L_total
L_GRPO
L_frame
L_slot_frame
L_conf
L_sae
format_score
frame_score
mean_slot_conf
```

如果使用 TensorBoard、Weights & Biases 或类似工具，建议同步记录上述指标，方便观察 loss 是否发散、format reward 是否掉到 0、confidence 是否整体塌缩。

## 10. Eval 流程

eval 时不能使用 GT label 构造 response。

正确流程：

1. 输入视频和规则 prompt
2. 模型自己生成 response
3. 从生成 response 中解析：

```text
<think>
Violated rules:
<RULE> tokens
```

4. 若 `<RULE>` 数量不足 4，则该样本记为格式错误或补零
5. 用生成的 `<RULE>` hidden states 走后续检测路径
6. 得到帧级异常分数：

```text
slot_frame_score_k[t] = sigmoid(det_logits_k[t])
weighted_slot_score_k[t] = sigmoid(conf_logits[k]) * slot_frame_score_k[t]
frame_score[t] = max_k weighted_slot_score_k[t]
```

7. 和 frame-level GT 计算：

- frame AUC
- frame AP
- frame F1
- temporal IoU
- clip-level accuracy

clip-level score 可由 frame score 聚合：

```text
clip_score = max_t frame_score[t]
```

## 11. 关键注意事项

### 11.1 不要用 noisy-OR 聚合未校准 slot

如果每个 slot 初始概率约为 0.5，noisy-OR 会得到：

```text
1 - 0.5^4 = 0.9375
```

这会导致所有 clip 都被判异常。

对于 clip-level score，推荐：

```text
clip_score = max_t max_k sigmoid(det_logits_k[t])
```

而不是直接对 slot confidence 做 noisy-OR。

### 11.2 `<RULE>` embedding 必须保存

训练时新增的 `<RULE>` embedding 必须随 checkpoint 保存。

eval 时必须恢复同一个 embedding，不能重新随机初始化。

### 11.3 confidence 需要 frame-level label 才稳定

仅有 video-level label 时，`conf_logits` 缺少明确 target。

有 frame-level label 后，才能通过：

```text
slot ↔ abnormal segment matching
```

构造稳定的 `conf_target`。

### 11.4 eval 必须使用模型生成 response

不能使用 GT label 构造：

```text
<think>...</think>
Violated rules:<RULE> <RULE> <RULE> <RULE>
```

否则会发生 label leakage。

### 11.5 K=4 是最大 slot 数，不是固定异常数

```text
K=4
```

表示最多 4 个候选异常解释/规则 slot。

如果一个 clip 只有一个异常片段，则通常只有一个 slot 的 confidence target 为 1，其余为 0。

如果没有异常片段，则 4 个 slot 的 confidence target 都为 0。

## 12. 推荐默认超参

```text
K = 4
SAE hook layer = 根据 baseline 模型选择
num_train_frames_per_clip = 12
λ_frame = 1.0
λ_conf = 0.2
λ_sae = 0.05
λ_dice = 1.0
GRPO format weight α = 0.3
GRPO frame/task weight β = 0.7
rollout_n = 4
cliprange = 0.2
base_lr = 1.6e-6
weight_decay = 1.0e-2
max_grad_norm = 1.0
lr_qb_rate = 30
lr_head_rate = 25
lr_pe_rate = 10
lr_decoder_rate = 5
```

学习率按 SegCompass 的 param group 思路设置：

```text
query_book lr = base_lr * lr_qb_rate
head lr = base_lr * lr_head_rate
pos/RULE embedding lr = base_lr * lr_pe_rate
SAE lr = base_lr * lr_decoder_rate
```

confidence head 最后一层 bias 建议初始化为负值：

```python
nn.init.constant_(conf_head[-1].bias, -2.0)
```

避免初始阶段所有 slot 都高置信。

## 13. 期望输出文件

训练输出：

```text
checkpoints/
metrics.jsonl
train_config.json
```

eval 输出：

```text
predictions.jsonl
frame_scores.npy
summary.json
```

`predictions.jsonl` 建议包含：

```json
{
  "clip_id": "...",
  "label_clip": 1,
  "frame_labels": [0, 0, 1, 1, 0],
  "frame_scores": [0.1, 0.2, 0.8, 0.7, 0.2],
  "slot_frame_preds": [[0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
  "slot_conf": [0.9, 0.1, 0.0, 0.0],
  "format_score": 1.0
}
```
