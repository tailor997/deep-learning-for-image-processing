import torch
import torchvision.models as models


def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = models.resnet34(pretrained=True)  # 加载预训练的resnet34模型
    model.load_state_dict(torch.load(checkpoint))  # 加载你自己的权重（如果有的话）
    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    checkpoint = './resnet34-pre.pth'
    onnx_path = './resnet34-pre.onnx'
    input = torch.randn(1, 3, 224, 224)  # 更改输入张量的大小和通道数
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)
