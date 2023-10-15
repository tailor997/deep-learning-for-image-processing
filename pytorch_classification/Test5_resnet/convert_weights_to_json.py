import torch
import json

# 加载.pth文件
state_dict = torch.load('./resNet34_flower_cls5.pth')

# 将权重数据从状态字典提取，并将其保存为字典
weights_dict = {key: value.tolist() for key, value in state_dict.items()}

# 保存为JSON文件
with open('model_weights.json', 'w') as json_file:
    json.dump(weights_dict, json_file)
