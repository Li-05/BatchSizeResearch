import os
import torch
import torch.optim as optim
import torch.nn as nn
import json
import numpy as np

def gauss_noise(model, iter, noise_std=0.1, top_Percent_weights=0.2):
    if(iter<=100):
        noise_std = noise_std/iter
    else:
        return
    print("add gauss noise")
    for name, param in model.named_parameters():
        if 'weight' in name:  # 只对权重参数添加噪声
            device = param.device  # 获取参数所在的设备
            weight_values = param.data.cpu().numpy()
            weight_abs = np.abs(weight_values)
            
            top_N_weights = int(np.ceil(weight_abs.size * top_Percent_weights))

            # 使用 unravel_index 获取多维索引
            top_N_indices = np.unravel_index(np.argsort(weight_abs, axis=None)[-top_N_weights:], weight_abs.shape)

            noise = np.random.normal(loc=0.0, scale=noise_std, size=top_N_weights)
            weight_values[top_N_indices] += noise
            param.data = torch.from_numpy(weight_values).to(device)