import torch

# 加载.pth文件
# state_dict = torch.load('your_model.pth')
state_dict = torch.load('./resNet34_flower_cls5.pth')


# 打印模型状态字典的内容
print(state_dict)
