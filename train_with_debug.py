from xtuner.tools.train import main as train
try:
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass

# 在.vscode下的launch.json中设置配置文件，然后可以用vscode的debug启动该程序
# 如果需要直接运行，不使用debug, 启动命令参考scripts/train_multi_gpu.sh文件中的命令
if __name__ == '__main__':
    train()