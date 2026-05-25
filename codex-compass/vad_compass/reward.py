import re

import torch
import torch.nn.functional as F


def format_score(response, pos_token="<ANOM_POS>", k_slots=4):
    """SegCompass-style format score for VAD responses."""
    think_blocks = list(re.finditer(r"<think>(.*?)</think>", response, flags=re.DOTALL))
    if len(think_blocks) != 1:
        return 0.0
    think = think_blocks[0]
    content = think.group(1)
    if not re.search(r"\S", content or ""):
        return 0.0
    if response.count(pos_token) != int(k_slots):
        return 0.0
    answers = re.findall(r"Answer:\s*(normal|abnormal)", response, flags=re.IGNORECASE)
    if len(answers) != 1:
        return 0.0
    answer_idx = response.lower().find("answer:")
    if answer_idx < think.end():
        return 0.0
    last_pos_idx = response.rfind(pos_token)
    if last_pos_idx < think.end() or answer_idx < last_pos_idx:
        return 0.0
    long_think = len(content) > 2048
    has_non_ws_before = bool(re.search(r"\S", response[: think.start()]))
    first_pos_idx = response.find(pos_token)
    between = response[think.end() : first_pos_idx]
    has_long_between = len(between.split()) > 10
    return 0.9 if (long_think or has_non_ws_before or has_long_between) else 1.0


def answer_from_action(action):
    return "abnormal" if int(action) == 1 else "normal"


def synthetic_response(label, action, abnormal_prob, pos_token="<ANOM_POS>", k_slots=4):
    answer = answer_from_action(action)
    conf = abnormal_prob if action == 1 else 1.0 - abnormal_prob
    pos_tokens = " ".join([pos_token] * int(k_slots))
    return (
        "<think> The video-level evidence is evaluated from sparse visual "
        f"concept slots. The predicted class is {answer} with confidence {conf:.3f}. "
        "</think>\n"
        f"Here are the {int(k_slots)} anomaly concentration tokens:\n"
        f"{pos_tokens}\n"
        f"Answer: {answer}"
    )


def vad_reward(label, action, response, format_weight=0.3, task_weight=0.7, pos_token="<ANOM_POS>", k_slots=4):
    fs = format_score(response, pos_token=pos_token, k_slots=k_slots)
    correct = 1.0 if int(action) == int(label) else 0.0
    return format_weight * fs + task_weight * correct, fs, correct


def grpo_advantages(rewards, group_size, eps=1e-6):
    rewards = rewards.view(-1, group_size)
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    adv = (rewards - mean) / (std + eps)
    return adv.reshape(-1)


def bernoulli_grpo_loss(logits, actions, old_log_probs, advantages, cliprange=0.2):
    dist = torch.distributions.Bernoulli(logits=logits)
    log_probs = dist.log_prob(actions.float())
    ratio = torch.exp(log_probs - old_log_probs)
    losses = -advantages * ratio
    losses_clipped = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    loss = torch.max(losses, losses_clipped).mean()
    approx_kl = (old_log_probs - log_probs).mean()
    clipfrac = (losses_clipped > losses).float().mean()
    return loss, approx_kl, clipfrac


def video_bce_loss(video_prob, labels):
    return F.binary_cross_entropy(video_prob.clamp(1e-6, 1.0 - 1e-6), labels.float())
