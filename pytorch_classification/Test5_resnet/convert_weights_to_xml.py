import torch
import xml.etree.ElementTree as ET

# 加载.pth文件
state_dict = torch.load('./resNet34_flower_cls5.pth')

# 创建XML根元素
root = ET.Element("model_weights")

# 将权重数据转换为XML格式
for key, value in state_dict.items():
    weight_element = ET.Element("weight")
    weight_element.set("name", key)
    weight_element.text = str(value.tolist())
    root.append(weight_element)

# 创建XML树
tree = ET.ElementTree(root)

# 保存XML文件
tree.write('model_weights.xml')

# 解析XML文件并查看权重数据
tree = ET.parse('model_weights.xml')
root = tree.getroot()

for weight_element in root:
    name = weight_element.get("name")
    weight_data = weight_element.text
    print(f"Parameter Name: {name}")
    print(f"Weight Data: {weight_data}")
