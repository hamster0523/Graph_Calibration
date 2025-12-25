from typing import List, Dict, Any
from verl import DataProto
import sys
import os
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import roc_auc_score

def kl_divergence(p, q):
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)  # 也 clip p，避免 0*log(0/⋯) 数值问题
    q = np.clip(q, eps, 1.0)
    p /= p.sum()  # 以防万一
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def calculate_rms_and_geometric_mean(values):
    values = list(map(float, values))
    n = len(values)
    mean = sum(values) / n
    rms = math.sqrt(sum(x**2 for x in values) / n)

    # 几何均值要防 0
    eps = 1e-12
    safe_vals = [max(v, eps) for v in values]
    geometric_mean = math.exp(sum(math.log(v) for v in safe_vals) / n)

    max_val = max(values)
    return mean, rms, geometric_mean, max_val

def calculate_kl_weighted(step_mean_confidence, forward_marginals, ref_marginals):
    weighted_confidences, total_weight = [], 0.0
    for i in range(len(step_mean_confidence)):
        kS, kA, kO = f"S_{i}", f"A_{i}", f"O_{i}"
        if kS not in forward_marginals or kA not in forward_marginals or kO not in forward_marginals:
            continue
        if kS not in ref_marginals or kA not in ref_marginals or kO not in ref_marginals:
            continue

        kl1 = kl_divergence(forward_marginals[kS], ref_marginals[kS])
        kl2 = kl_divergence(forward_marginals[kA], ref_marginals[kA])
        kl3 = kl_divergence(forward_marginals[kO], ref_marginals[kO])
        kl_score = (kl1 + kl2 + kl3) / 3.0

        # 防溢出
        weight = math.exp(min(kl_score, 10.0))
        weighted_confidences.append(step_mean_confidence[i] * weight)
        total_weight += weight

    if total_weight <= 0:
        return float(np.mean(step_mean_confidence))
    return float(np.sum(weighted_confidences) / total_weight)
 
def get_confidence(main_marginals: Dict[str, Any], ref_marginals: Dict[str, Any]):
    steps = []
    i = 0
    while True:
        kS, kA, kO = f"S_{i}", f"A_{i}", f"O_{i}"
        if kS in main_marginals and kA in main_marginals and kO in main_marginals:
            s = float(main_marginals[kS][1])
            a = float(main_marginals[kA][1])
            o = float(main_marginals[kO][1])
            steps.append((s, a, o))
            i += 1
        else:
            break

    if not steps:
        # 兜底：尽力从全部键里拼
        state_confidences, action_confidences, observation_confidences = [], [], []
        for name, marg in main_marginals.items():
            if isinstance(marg, (list, tuple)) and len(marg) >= 2:
                p1 = float(marg[1])
                if name.startswith("S_"):
                    state_confidences.append(p1)
                elif name.startswith("A_"):
                    action_confidences.append(p1)
                elif name.startswith("O_"):
                    observation_confidences.append(p1)
        steps = list(zip(state_confidences, action_confidences, observation_confidences))

    step_mean_confidence = [(s + a + o) / 3.0 for (s, a, o) in steps]
    mean, rms, geo_mean, max_val = calculate_rms_and_geometric_mean(step_mean_confidence)

    # KL权重：主分布(main_marginals) vs 参考分布(ref_marginals)
    kl_weighted = calculate_kl_weighted(step_mean_confidence, main_marginals, ref_marginals)
    return mean, rms, geo_mean, max_val, kl_weighted
        
def calculate_ece(mean_list, rms_list, gmean_list, max_list, kl_weighted_list, y_true):
    """
    计算ECE指标
    mean_list: 各样本的mean置信度列表
    rms_list: 各样本的rms置信度列表
    gmean_list: 各样本的geometric mean置信度列表
    max_list: 各样本的max置信度列表
    kl_weighted_list: 各样本的kl_weighted置信度列表
    y_true: 各样本的真实标签列表（1表示正确，0表示错误）
    
    返回各指标的ECE值
    """
    mean_ece = ece(mean_list, y_true)
    rms_ece = ece(rms_list, y_true)
    gmean_ece = ece(gmean_list, y_true)
    max_ece = ece(max_list, y_true)
    kl_weighted_ece = ece(kl_weighted_list, y_true)
    
    return (mean_ece, rms_ece, gmean_ece, max_ece, kl_weighted_ece)

def calculate_ece_verbal(mean_list, rms_list, gmean_list, max_list, y_true):
    """
    计算ECE指标
    mean_list: 各样本的mean置信度列表
    rms_list: 各样本的rms置信度列表
    gmean_list: 各样本的geometric mean置信度列表
    max_list: 各样本的max置信度列表
    y_true: 各样本的真实标签列表（1表示正确，0表示错误）
    
    返回各指标的ECE值
    """
    mean_ece = ece(mean_list, y_true)
    rms_ece = ece(rms_list, y_true)
    gmean_ece = ece(gmean_list, y_true)
    max_ece = ece(max_list, y_true)
    
    return (mean_ece, rms_ece, gmean_ece, max_ece)

def calculate_aucroc(mean_list, rms_list, gmean_list, max_list, kl_weighted_list, y_true):
    """
    计算AUC-ROC指标
    mean_list: 各样本的mean置信度列表
    rms_list: 各样本的rms置信度列表
    gmean_list: 各样本的geometric mean置信度列表
    max_list: 各样本的max置信度列表
    kl_weighted_list: 各样本的kl_weighted置信度列表
    y_true: 各样本的真实标签列表（1表示正确，0表示错误）
    
    返回各指标的AUC-ROC值
    """
    def safe_roc_auc_score(y_true, y_scores):
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return float('nan')
    
    mean_auc = safe_roc_auc_score(y_true, mean_list)
    rms_auc = safe_roc_auc_score(y_true, rms_list)
    gmean_auc = safe_roc_auc_score(y_true, gmean_list)
    max_auc = safe_roc_auc_score(y_true, max_list)
    kl_weighted_auc = safe_roc_auc_score(y_true, kl_weighted_list)
    
    return (mean_auc, rms_auc, gmean_auc, max_auc, kl_weighted_auc)    

def calculate_aucroc_verbal(mean_list, rms_list, gmean_list, max_list, y_true):
    """
    计算AUC-ROC指标
    mean_list: 各样本的mean置信度列表
    rms_list: 各样本的rms置信度列表
    gmean_list: 各样本的geometric mean置信度列表
    max_list: 各样本的max置信度列表
    y_true: 各样本的真实标签列表（1表示正确，0表示错误）
    
    返回各指标的AUC-ROC值
    """
    def safe_roc_auc_score(y_true, y_scores):
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return float('nan')
    
    mean_auc = safe_roc_auc_score(y_true, mean_list)
    rms_auc = safe_roc_auc_score(y_true, rms_list)
    gmean_auc = safe_roc_auc_score(y_true, gmean_list)
    max_auc = safe_roc_auc_score(y_true, max_list)
    
    return (mean_auc, rms_auc, gmean_auc, max_auc)

def ece(confidences, labels, num_bins=10):
    """Expected Calibration Error"""
    if not confidences:
        return 0.0

    bin_size = 1.0 / num_bins
    ece_value = 0.0
    n = len(confidences)

    for i in range(num_bins):
        bin_lower = i * bin_size
        # 最后一个桶包含 1.0
        bin_upper = 1.0 if i == num_bins - 1 else (i + 1) * bin_size

        bin_indices = [j for j, conf in enumerate(confidences)
                       if (conf >= bin_lower and (conf < bin_upper or (i == num_bins - 1 and conf <= bin_upper)))]

        if not bin_indices:
            continue

        bin_confidences = [confidences[j] for j in bin_indices]
        bin_labels = [labels[j] for j in bin_indices]

        avg_confidence = sum(bin_confidences) / len(bin_confidences)
        accuracy = sum(bin_labels) / len(bin_labels)
        ece_value += (len(bin_indices) / n) * abs(avg_confidence - accuracy)

    return float(ece_value)

def get_verbal_confidence(trajectory: Dict[str, Any]):
    steps = trajectory.get("steps", [])
    verbal_confidences = []
    for step in steps:
        conf_text = step.get("action_verbal_confidence", "")
        try:
            conf_value = float(conf_text)
            verbal_confidences.append(conf_value)
        except (ValueError, TypeError):
            continue  

    if not verbal_confidences:
        return 0.0, 0.0, 0.0, 0.0  

    mean, rms, geo_mean, max_val = calculate_rms_and_geometric_mean(verbal_confidences)
    return mean, rms, geo_mean, max_val

# def compute(data_batch: DataProto) -> DataProto:
#     batch_meta_info = data_batch.meta_info
#     #print(batch_meta_info)
#     batch_size = data_batch.batch.batch_size[0]
    
#     # 用于存储各类置信度的统计量
#     mean_forward, rms_forward, gmean_forward, max_forward, kl_weighted_forward = [], [], [], [], []
#     mean_backward, rms_backward, gmean_backward, max_backward, kl_weighted_backward = [], [], [], [], []
#     mean_verbals, rms_verbals, gmean_verbals, max_verbals = [], [], [], []
    
#     # 用于存储ECE的计算结果
#     mean_ece_forward, rms_ece_forward, gmean_ece_forward, max_ece_forward, kl_weighted_ece_forward = [], [], [], [], []
#     mean_ece_backward, rms_ece_backward, gmean_ece_backward, max_ece_backward, kl_weighted_ece_backward = [], [], [], [], []
#     mean_ece_verbal, rms_ece_verbal, gmean_ece_verbal, max_ece_verbal = [], [], [], []

#     # 用于计算AUC-ROC
#     mean_auc_forward, rms_auc_forward, gmean_auc_forward, max_auc_forward, kl_weighted_auc_forward = [], [], [], [], []
#     mean_auc_backward, rms_auc_backward, gmean_auc_backward, max_auc_backward, kl_weighted_auc_backward = [], [], [], [], []
#     mean_auc_verbal, rms_auc_verbal, gmean_auc_verbal, max_auc_verbal = [], [], [], []

#     true_counts = 0
    
#     y_true = []

#     for idx in range(batch_size):
#         #print(info)
#         graph_calibrations_re = batch_meta_info.get("graph_calibrations", [])[idx] 
#         #print(graph_calibrations_re)     
#         answer_state = graph_calibrations_re.get("answer_state")
#         if answer_state:
#             true_counts += 1
#         forward_marginals = graph_calibrations_re.get("forward_marginals")
#         #print(forward_marginals)
#         forward_no_evidence_marginals = graph_calibrations_re.get("forward_no_evidence_marginals")
#         #print(forward_no_evidence_marginals)
#         posterior_marginals = graph_calibrations_re.get("posterior_marginals")
#         #print(posterior_marginals)

#         # 获取前向置信度和后向置信度
#         mean, rms, geo_mean, max_val, kl_weighted = get_confidence(forward_marginals, forward_no_evidence_marginals)
#         mean_p, rms_p, geo_mean_p, max_val_p, kl_weighted_p = get_confidence(posterior_marginals, forward_no_evidence_marginals)        
        
#         # 如果你也想要 KL(posterior || forward) 的权重，可额外存一份：
#         # _, _, _, _, kl_post_vs_fwd = get_confidence(posterior_marginals, forward_marginals)
        
#         # 取前向的verbal confidence
#         trajectory = batch_meta_info.get("trajectories", [])[idx]
#         mean_verbal, rms_verbal, geo_mean_verbal, max_verbal = get_verbal_confidence(trajectory)
        
#         # 将结果存储
#         mean_forward.append(mean)
#         rms_forward.append(rms)
#         gmean_forward.append(geo_mean)
#         max_forward.append(max_val)
#         kl_weighted_forward.append(kl_weighted)
        
#         mean_verbals.append(mean_verbal)
#         rms_verbals.append(rms_verbal)
#         gmean_verbals.append(geo_mean_verbal)
#         max_verbals.append(max_verbal)
        
#         mean_backward.append(mean_p)
#         rms_backward.append(rms_p)
#         gmean_backward.append(geo_mean_p)
#         max_backward.append(max_val_p)
#         kl_weighted_backward.append(kl_weighted_p)
        
#         y_true.append(1 if answer_state else 0)  

#     # 计算ECE
#     mean_ece_forward, rms_ece_forward, gmean_ece_forward, max_ece_forward, kl_weighted_ece_forward = calculate_ece(
#         mean_forward, rms_forward, gmean_forward, max_forward, kl_weighted_forward, y_true
#     )

#     mean_ece_backward, rms_ece_backward, gmean_ece_backward, max_ece_backward, kl_weighted_ece_backward = calculate_ece(
#         mean_backward, rms_backward, gmean_backward, max_backward, kl_weighted_backward, y_true
#     )
    
#     mean_ece_verbal, rms_ece_verbal, gmean_ece_verbal, max_ece_verbal = calculate_ece_verbal(
#         mean_verbals, rms_verbals, gmean_verbals, max_verbals, y_true
#     )
        
#     # 计算AUC-ROC
#     mean_auc_forward, rms_auc_forward, gmean_auc_forward, max_auc_forward, kl_weighted_auc_forward = calculate_aucroc(
#         mean_forward, rms_forward, gmean_forward, max_forward, kl_weighted_forward, y_true
#     )

#     mean_auc_backward, rms_auc_backward, gmean_auc_backward, max_auc_backward, kl_weighted_auc_backward = calculate_aucroc(
#         mean_backward, rms_backward, gmean_backward, max_backward, kl_weighted_backward, y_true
#     )
#     mean_auc_verbal, rms_auc_verbal, gmean_auc_verbal, max_auc_verbal = calculate_aucroc_verbal(
#         mean_verbals, rms_verbals, gmean_verbals, max_verbals, y_true
#     )
    
#     # 将计算结果添加到 meta_info 中
#     meta_info = data_batch.meta_info
#     meta_info["mean_forward"] = mean_forward
#     meta_info["rms_forward"] = rms_forward
#     meta_info["gmean_forward"] = gmean_forward
#     meta_info["max_forward"] = max_forward
#     meta_info["kl_weighted_forward"] = kl_weighted_forward
    
#     meta_info["mean_verbal"] = mean_verbals
#     meta_info["rms_verbal"] = rms_verbals
#     meta_info["gmean_verbal"] = gmean_verbals
#     meta_info["max_verbal"] = max_verbals

#     meta_info["mean_backward"] = mean_backward
#     meta_info["rms_backward"] = rms_backward
#     meta_info["gmean_backward"] = gmean_backward
#     meta_info["max_backward"] = max_backward
#     meta_info["kl_weighted_backward"] = kl_weighted_backward
    
#     meta_info["mean_ece_forward"] = mean_ece_forward
#     meta_info["rms_ece_forward"] = rms_ece_forward
#     meta_info["gmean_ece_forward"] = gmean_ece_forward
#     meta_info["max_ece_forward"] = max_ece_forward
#     meta_info["kl_weighted_ece_forward"] = kl_weighted_ece_forward
    
#     meta_info['mean_ece_verbal'] = mean_ece_verbal
#     meta_info['rms_ece_verbal'] = rms_ece_verbal
#     meta_info['gmean_ece_verbal'] = gmean_ece_verbal
#     meta_info['max_ece_verbal'] = max_ece_verbal

#     meta_info["mean_ece_backward"] = mean_ece_backward
#     meta_info["rms_ece_backward"] = rms_ece_backward
#     meta_info["gmean_ece_backward"] = gmean_ece_backward
#     meta_info["max_ece_backward"] = max_ece_backward
#     meta_info["kl_weighted_ece_backward"] = kl_weighted_ece_backward
    
#     meta_info["mean_auc_forward"] = mean_auc_forward
#     meta_info["rms_auc_forward"] = rms_auc_forward
#     meta_info["gmean_auc_forward"] = gmean_auc_forward
#     meta_info["max_auc_forward"] = max_auc_forward
#     meta_info["kl_weighted_auc_forward"] = kl_weighted_auc_forward
    
#     meta_info['mean_auc_verbal'] = mean_auc_verbal
#     meta_info['rms_auc_verbal'] = rms_auc_verbal
#     meta_info['gmean_auc_verbal'] = gmean_auc_verbal
#     meta_info['max_auc_verbal'] = max_auc_verbal

#     meta_info["mean_auc_backward"] = mean_auc_backward
#     meta_info["rms_auc_backward"] = rms_auc_backward
#     meta_info["gmean_auc_backward"] = gmean_auc_backward
#     meta_info["max_auc_backward"] = max_auc_backward
#     meta_info["kl_weighted_auc_backward"] = kl_weighted_auc_backward
    
#     meta_info["acc"] = true_counts / batch_size if batch_size > 0 else 0.0 

#     # 创建新的 DataProto 对象并返回
#     new_data_batch = DataProto(
#         batch=data_batch.batch,
#         non_tensor_batch=data_batch.non_tensor_batch,
#         meta_info=meta_info
#     )

#     return new_data_batch

def compute(data_batch: DataProto) -> DataProto:
    """
    从 non_tensor_batch 里读取 graph_calibrations / trajectories，
    通过 uid 做对齐，计算各类置信度统计、ECE 和 AUC，
    然后把这些标量写回 meta_info，供 trainer 读 metrics。
    """

    # ---- 正确的数据来源：non_tensor_batch，而不是 meta_info ----
    non_tensor = getattr(data_batch, "non_tensor_batch", {}) or {}

    # ======== 1. 取 raw 数据：不要用 `or` 链接 ndarray！========
    if "graph_calibrations" in non_tensor:
        graph_cal_raw = non_tensor["graph_calibrations"]
    else:
        graph_cal_raw = []

    if "trajectories" in non_tensor:
        traj_raw = non_tensor["trajectories"]
    elif "hamster_trajectories" in non_tensor:
        traj_raw = non_tensor["hamster_trajectories"]
    else:
        traj_raw = []

    uid_raw = non_tensor.get("rollout_uid", None)

    # 统一成 Python list
    def to_list(x):
        if x is None:
            return []
        if hasattr(x, "tolist"):
            return x.tolist()
        return list(x)

    graph_cal_list = to_list(graph_cal_raw)
    trajectories_list = to_list(traj_raw)
    uid_list = [str(u) for u in to_list(uid_raw)]

    # ---- batch_size 建议以张量 batch 为准 ----
    if "responses" in data_batch.batch:
        batch_size = int(data_batch.batch["responses"].shape[0])
    else:
        first_tensor = next(iter(data_batch.batch.values()))
        batch_size = int(first_tensor.shape[0])

    # ===== 关键：构造 uid -> 对象 的映射 =====
    uid2cal = {}
    for obj in graph_cal_list:
        if isinstance(obj, dict):
            u = obj.get("rollout_uid", None)
            if u is not None:
                uid2cal[str(u)] = obj

    uid2traj = {}
    for obj in trajectories_list:
        if isinstance(obj, dict):
            u = obj.get("rollout_uid", None)
            if u is not None:
                uid2traj[str(u)] = obj

    # 统计用的列表
    mean_forward, rms_forward, gmean_forward, max_forward, kl_weighted_forward = [], [], [], [], []
    mean_backward, rms_backward, gmean_backward, max_backward, kl_weighted_backward = [], [], [], [], []

    mean_verbals, rms_verbals, gmean_verbals, max_verbals = [], [], [], []

    y_true = []
    true_counts = 0

    # ===== 逐样本（按当前 batch 顺序）通过 uid 匹配 =====
    for idx in range(batch_size):
        if idx >= len(uid_list):
            # 没有 uid，保守跳过
            continue
        uid = uid_list[idx]

        graph_calibrations_re = uid2cal.get(uid, None)
        if not isinstance(graph_calibrations_re, dict):
            # 没有对应的 graph_calibration，也跳过
            continue

        answer_state = graph_calibrations_re.get("answer_state")
        if answer_state:
            true_counts += 1

        forward_marginals = graph_calibrations_re.get("forward_marginals") or {}
        forward_no_evidence_marginals = graph_calibrations_re.get("forward_no_evidence_marginals") or {}
        posterior_marginals = graph_calibrations_re.get("posterior_marginals") or {}

        # 如果缺 forward / posterior / no-evidence，说明该样本图推理没跑成功，也跳过
        if not forward_marginals or not forward_no_evidence_marginals or not posterior_marginals:
            continue

        # ---- 前向置信度 & 后向置信度 ----
        mean_f, rms_f, geo_mean_f, max_val_f, kl_weighted_f = get_confidence(
            forward_marginals, forward_no_evidence_marginals
        )
        mean_b, rms_b, geo_mean_b, max_val_b, kl_weighted_b = get_confidence(
            posterior_marginals, forward_no_evidence_marginals
        )

        # ---- verbal confidence，从对应的 trajectory 里拿（同样通过 uid 匹配）----
        traj = uid2traj.get(uid, None)
        if isinstance(traj, dict):
            mean_verbal, rms_verbal, geo_mean_verbal, max_verbal = get_verbal_confidence(traj)
        else:
            mean_verbal = rms_verbal = geo_mean_verbal = max_verbal = 0.0

        # 存储
        mean_forward.append(mean_f)
        rms_forward.append(rms_f)
        gmean_forward.append(geo_mean_f)
        max_forward.append(max_val_f)
        kl_weighted_forward.append(kl_weighted_f)

        mean_backward.append(mean_b)
        rms_backward.append(rms_b)
        gmean_backward.append(geo_mean_b)
        max_backward.append(max_val_b)
        kl_weighted_backward.append(kl_weighted_b)

        mean_verbals.append(mean_verbal)
        rms_verbals.append(rms_verbal)
        gmean_verbals.append(geo_mean_verbal)
        max_verbals.append(max_verbal)

        y_true.append(1 if answer_state else 0)

    # 如果一个样本都没成功取到，就直接把 acc 设 0，其他指标也设 0
    meta_info = data_batch.meta_info

    if not y_true:
        meta_info["mean_forward"] = []
        meta_info["rms_forward"] = []
        meta_info["gmean_forward"] = []
        meta_info["max_forward"] = []
        meta_info["kl_weighted_forward"] = []

        meta_info["mean_verbal"] = []
        meta_info["rms_verbal"] = []
        meta_info["gmean_verbal"] = []
        meta_info["max_verbal"] = []

        meta_info["mean_backward"] = []
        meta_info["rms_backward"] = []
        meta_info["gmean_backward"] = []
        meta_info["max_backward"] = []
        meta_info["kl_weighted_backward"] = []

        meta_info["mean_ece_forward"] = 0.0
        meta_info["rms_ece_forward"] = 0.0
        meta_info["gmean_ece_forward"] = 0.0
        meta_info["max_ece_forward"] = 0.0
        meta_info["kl_weighted_ece_forward"] = 0.0

        meta_info["mean_ece_verbal"] = 0.0
        meta_info["rms_ece_verbal"] = 0.0
        meta_info["gmean_ece_verbal"] = 0.0
        meta_info["max_ece_verbal"] = 0.0

        meta_info["mean_ece_backward"] = 0.0
        meta_info["rms_ece_backward"] = 0.0
        meta_info["gmean_ece_backward"] = 0.0
        meta_info["max_ece_backward"] = 0.0
        meta_info["kl_weighted_ece_backward"] = 0.0

        meta_info["mean_auc_forward"] = 0.0
        meta_info["rms_auc_forward"] = 0.0
        meta_info["gmean_auc_forward"] = 0.0
        meta_info["max_auc_forward"] = 0.0
        meta_info["kl_weighted_auc_forward"] = 0.0

        meta_info["mean_auc_verbal"] = 0.0
        meta_info["rms_auc_verbal"] = 0.0
        meta_info["gmean_auc_verbal"] = 0.0
        meta_info["max_auc_verbal"] = 0.0

        meta_info["mean_auc_backward"] = 0.0
        meta_info["rms_auc_backward"] = 0.0
        meta_info["gmean_auc_backward"] = 0.0
        meta_info["max_auc_backward"] = 0.0
        meta_info["kl_weighted_auc_backward"] = 0.0

        meta_info["acc"] = 0.0

        return DataProto(
            batch=data_batch.batch,
            non_tensor_batch=data_batch.non_tensor_batch,
            meta_info=meta_info,
        )

    # ===== 下面 ECE & AUC 计算逻辑保持不变，只是用上面整理好的 list =====

    mean_ece_forward, rms_ece_forward, gmean_ece_forward, max_ece_forward, kl_weighted_ece_forward = calculate_ece(
        mean_forward, rms_forward, gmean_forward, max_forward, kl_weighted_forward, y_true
    )

    mean_ece_backward, rms_ece_backward, gmean_ece_backward, max_ece_backward, kl_weighted_ece_backward = calculate_ece(
        mean_backward, rms_backward, gmean_backward, max_backward, kl_weighted_backward, y_true
    )

    mean_ece_verbal, rms_ece_verbal, gmean_ece_verbal, max_ece_verbal = calculate_ece_verbal(
        mean_verbals, rms_verbals, gmean_verbals, max_verbals, y_true
    )

    mean_auc_forward, rms_auc_forward, gmean_auc_forward, max_auc_forward, kl_weighted_auc_forward = calculate_aucroc(
        mean_forward, rms_forward, gmean_forward, max_forward, kl_weighted_forward, y_true
    )

    mean_auc_backward, rms_auc_backward, gmean_auc_backward, max_auc_backward, kl_weighted_auc_backward = calculate_aucroc(
        mean_backward, rms_backward, gmean_backward, max_backward, kl_weighted_backward, y_true
    )

    mean_auc_verbal, rms_auc_verbal, gmean_auc_verbal, max_auc_verbal = calculate_aucroc_verbal(
        mean_verbals, rms_verbals, gmean_verbals, max_verbals, y_true
    )

    # ===== 写回 meta_info，供 trainer 和 logger 用 =====
    meta_info["mean_forward"] = mean_forward
    meta_info["rms_forward"] = rms_forward
    meta_info["gmean_forward"] = gmean_forward
    meta_info["max_forward"] = max_forward
    meta_info["kl_weighted_forward"] = kl_weighted_forward

    meta_info["mean_verbal"] = mean_verbals
    meta_info["rms_verbal"] = rms_verbals
    meta_info["gmean_verbal"] = gmean_verbals
    meta_info["max_verbal"] = max_verbals

    meta_info["mean_backward"] = mean_backward
    meta_info["rms_backward"] = rms_backward
    meta_info["gmean_backward"] = gmean_backward
    meta_info["max_backward"] = max_backward
    meta_info["kl_weighted_backward"] = kl_weighted_backward

    meta_info["mean_ece_forward"] = mean_ece_forward
    meta_info["rms_ece_forward"] = rms_ece_forward
    meta_info["gmean_ece_forward"] = gmean_ece_forward
    meta_info["max_ece_forward"] = max_ece_forward
    meta_info["kl_weighted_ece_forward"] = kl_weighted_ece_forward

    meta_info["mean_ece_verbal"] = mean_ece_verbal
    meta_info["rms_ece_verbal"] = rms_ece_verbal
    meta_info["gmean_ece_verbal"] = gmean_ece_verbal
    meta_info["max_ece_verbal"] = max_ece_verbal

    meta_info["mean_ece_backward"] = mean_ece_backward
    meta_info["rms_ece_backward"] = rms_ece_backward
    meta_info["gmean_ece_backward"] = gmean_ece_backward
    meta_info["max_ece_backward"] = max_ece_backward
    meta_info["kl_weighted_ece_backward"] = kl_weighted_ece_backward

    meta_info["mean_auc_forward"] = mean_auc_forward
    meta_info["rms_auc_forward"] = rms_auc_forward
    meta_info["gmean_auc_forward"] = gmean_auc_forward
    meta_info["max_auc_forward"] = max_auc_forward
    meta_info["kl_weighted_auc_forward"] = kl_weighted_auc_forward

    meta_info["mean_auc_verbal"] = mean_auc_verbal
    meta_info["rms_auc_verbal"] = rms_auc_verbal
    meta_info["gmean_auc_verbal"] = gmean_auc_verbal
    meta_info["max_auc_verbal"] = max_auc_verbal

    meta_info["mean_auc_backward"] = mean_auc_backward
    meta_info["rms_auc_backward"] = rms_auc_backward
    meta_info["gmean_auc_backward"] = gmean_auc_backward
    meta_info["max_auc_backward"] = max_auc_backward
    meta_info["kl_weighted_auc_backward"] = kl_weighted_auc_backward

    # acc 用成功样本数 / 有效样本数
    meta_info["acc"] = true_counts / len(y_true) if len(y_true) > 0 else 0.0

    return DataProto(
        batch=data_batch.batch,
        non_tensor_batch=data_batch.non_tensor_batch,
        meta_info=meta_info,
    )
