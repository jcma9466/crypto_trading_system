import os
import sys

import numpy as np
import torch as th
from torch.nn.utils import clip_grad_norm_
from typing import Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.seq_data import ConfigData, convert_btc_csv_to_btc_npy

TEN = th.Tensor


def check_mamba_availability():
    """检查mamba是否可用"""
    try:
        from mamba_ssm import Mamba
        return True
    except ImportError:
        print("Warning: mamba_ssm not available, falling back to RNN model")
        return False


def get_mamba_model_paths(args: ConfigData):
    """获取mamba模型相关路径"""
    data_dir = args.data_dir
    return {
        'net_path': f"{data_dir}/BTC_15m_mamba_predict.pth",
        'predict_path': f"{data_dir}/BTC_15m_mamba_predict.npy",
        'flag_path': f"{data_dir}/mamba_trained.flag"
    }


def train_mamba_model(gpu_id: int):
    """训练Mamba模型"""
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

    '''Config for training'''
    batch_size = 128
    seq_len = 2 ** 8
    mid_dim = 128
    num_layers = 4
    
    epoch = 2 ** 10
    learning_rate = 5e-4
    weight_decay = 1e-4
    wup_dim = 64
    valid_gap = 32
    num_patience = 16
    clip_grad_norm = 5
    out_dir = './output'
    if_report = True

    '''data'''
    args = ConfigData()
    seq_data = SeqData(args=args, train_ratio=0.8)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim
    
    # Move data tensors to device to avoid CUDA errors
    seq_data.train_input_seq = seq_data.train_input_seq.to(device)
    seq_data.train_label_seq = seq_data.train_label_seq.to(device)
    seq_data.valid_input_seq = seq_data.valid_input_seq.to(device)
    seq_data.valid_label_seq = seq_data.valid_label_seq.to(device)
    
    mamba_paths = get_mamba_model_paths(args)

    '''Model'''
    from seq_net import MambaTCNRegNet
    net = MambaTCNRegNet(inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers).to(device)
    optimizer = th.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = th.nn.MSELoss(reduction='none')

    '''Record'''
    from seq_record import Evaluator
    evaluator = Evaluator(out_dir=out_dir)
    
    valid_batches = max(1, int(seq_data.valid_seq_len / seq_len / batch_size))
    print(f"| Mamba Validation batches: {valid_batches}")

    train_times = int(seq_data.train_seq_len / seq_len / batch_size * epoch)
    print(f"| Mamba train_seq_len {seq_data.train_seq_len}  train_times {train_times}")
    
    for step_idx in range(train_times):
        th.set_grad_enabled(True)
        net.train()
        inp, lab = seq_data.sample_for_train(batch_size=batch_size, seq_len=seq_len, device=device)
        out, _ = net(inp)
        obj = net.get_obj_value(criterion=criterion, out=out, lab=lab, wup_dim=wup_dim)
        _update_network(optimizer, obj.mean(), clip_grad_norm)

        evaluator.update_obj_train(obj=obj)

        if (step_idx % valid_gap == 0) or (step_idx == train_times - 1):
            th.set_grad_enabled(False)
            evaluator.update_obj_train(obj=None)

            '''update_obj_valid'''
            net.eval()
            valid_objs = []
            for _ in range(valid_batches):
                inp, lab = seq_data.sample_for_train(batch_size=batch_size, seq_len=seq_len, device=device)
                out, _ = net(inp)
                
                seq_len_ = min(out.shape[0], lab.shape[0])
                valid_objs.append(criterion(out[wup_dim:seq_len_], lab[wup_dim:seq_len_]))

            if valid_objs:
                valid_obj = th.cat(valid_objs, dim=0)
                evaluator.update_obj_valid(valid_obj)
            else:
                print("| Warning: Empty validation batch")
                evaluator.update_obj_valid(th.zeros(0))

            del inp, lab, out, valid_objs
            evaluator.update_obj_valid(obj=None)

            evaluator.log_print(step_idx=step_idx)
            evaluator.draw_train_valid_loss_curve(gpu_id=gpu_id)

            if evaluator.patience > num_patience:
                break
            if evaluator.patience == 0:
                best_valid_loss = evaluator.best_valid_loss
                th.save(net.state_dict(), f'{out_dir}/mamba_net_{step_idx:06}_{best_valid_loss:06.3f}.pth')

    # 保存mamba模型
    th.save(net.state_dict(), mamba_paths['net_path'])
    print(f'| save mamba network in {mamba_paths["net_path"]}')
    
    # 创建训练完成标记
    with open(mamba_paths['flag_path'], 'w') as f:
        f.write('mamba_trained')
    print(f'| mamba training completed, flag saved to {mamba_paths["flag_path"]}')

    # 生成预测结果
    predict_ary = np.empty_like(seq_data.valid_label_seq.cpu().numpy())
    hid: Optional[TEN] = None

    print(f"| mamba valid_seq_len {seq_data.valid_seq_len}  valid_times {seq_data.valid_seq_len // seq_len}")
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1]
        out, hid = net.forward(inp[:, None, :], hid)
        if out is not None:
            predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)
    
    np.save(mamba_paths['predict_path'], predict_ary)
    print(f'| save mamba predict in {mamba_paths["predict_path"]}')


class SeqData:
    def __init__(self, args: ConfigData, train_ratio: float = 0.8):
        input_ary_path = args.input_ary_path
        label_ary_path = args.label_ary_path

        '''Load or generate data'''
        if not os.path.exists(label_ary_path) or not os.path.exists(input_ary_path):
            convert_btc_csv_to_btc_npy(args=args)

        input_seq = np.load(input_ary_path)
        input_seq = np.nan_to_num(input_seq, nan=0.0, neginf=0.0, posinf=0.0)
        input_seq = th.tensor(input_seq, dtype=th.float32)
        assert th.isnan(input_seq).sum() == 0

        label_seq = np.load(label_ary_path)
        label_seq = np.nan_to_num(label_seq, nan=0.0, neginf=0.0, posinf=0.0)
        label_seq = th.tensor(label_seq, dtype=th.float32)
        assert th.isnan(label_seq).sum() == 0

        seq_len = label_seq.shape[0]
        self.input_dim = input_seq.shape[1]
        self.label_dim = label_seq.shape[1]

        seq_i0 = 0
        seq_i1 = int(seq_len * train_ratio)
        seq_i2 = int(seq_len - 1024) if 0 < train_ratio < 1 else seq_len
        self.train_seq_len = seq_i1 - seq_i0
        self.valid_seq_len = seq_i2 - seq_i1

        self.train_input_seq = input_seq[seq_i0:seq_i1]
        self.valid_input_seq = input_seq[seq_i1:seq_i2]
        self.train_label_seq = label_seq[seq_i0:seq_i1]
        self.valid_label_seq = label_seq[seq_i1:seq_i2]

    def sample_for_train(self, batch_size: int = 32, seq_len: int = 4096, device: th.device = th.device('cpu')):
        i0s = np.random.randint(1024, self.train_seq_len - seq_len, size=batch_size)
        ids = np.arange(seq_len)[None, :].repeat(batch_size, axis=0) + i0s[:, None]

        input_ary = self.train_input_seq[ids, :].permute(1, 0, 2)
        label_seq = self.train_label_seq[ids, :].permute(1, 0, 2)
        
        # Only move to device if not already on the correct device
        if input_ary.device != device:
            input_ary = input_ary.to(device)
        if label_seq.device != device:
            label_seq = label_seq.to(device)
            
        return input_ary, label_seq


def _update_network(optimizer, obj, clip_grad_norm):
    optimizer.zero_grad()
    obj.backward()
    clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=clip_grad_norm)
    optimizer.step()


def train_model(gpu_id: int):
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    
    '''data'''
    args = ConfigData()
    
    # 检查mamba可用性和是否已有mamba结果
    mamba_available = check_mamba_availability()
    mamba_paths = get_mamba_model_paths(args)
    
    if mamba_available:
        # 检查是否已有mamba训练结果
        if os.path.exists(mamba_paths['flag_path']) and os.path.exists(mamba_paths['net_path']):
            print("| Mamba model already trained, skipping training")
            return
        else:
            print("| Mamba available, using Mamba for training")
            train_mamba_model(gpu_id)
            return
    
    print("| Mamba not available, using RNN model")
    
    '''Config for training'''
    # 由于因子数减少，适当调整模型参数
    batch_size = 128  # 减小批次大小
    seq_len = 2 ** 8  
    mid_dim = 128  # 减小模型容量
    num_layers = 4  # 减少网络层数
    
    # 调整训练参数
    epoch = 2 ** 10  # 增加训练轮次
    learning_rate = 5e-4  # 调小学习率
    weight_decay = 1e-4  # 增加正则化强度
    wup_dim = 64  # 增加预热维度
    valid_gap = 32  # 更频繁地验证
    num_patience = 16  # 增加容忍度
    weight_decay = 1e-5  # 适当调整正则化强度
    learning_rate = 1e-3  # 增大学习率
    clip_grad_norm = 5  # 增加梯度裁剪阈值
    out_dir = './output'
    if_report = True

    seq_data = SeqData(args=args, train_ratio=0.8)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim

    '''Model'''
    from seq_net import RnnRegNet
    net = RnnRegNet(inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers).to(device)
    optimizer = th.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = th.nn.MSELoss(reduction='none')

    '''Record'''
    from seq_record import Evaluator
    evaluator = Evaluator(out_dir=out_dir)
    
    # Add validation batch count calculation
    valid_batches = max(1, int(seq_data.valid_seq_len / seq_len / batch_size))
    print(f"| Validation batches: {valid_batches}")

    seq_len = 2 ** 8
    train_times = int(seq_data.train_seq_len / seq_len / batch_size * epoch)
    print(f"| train_seq_len {seq_data.train_seq_len}  train_times {train_times}")
    for step_idx in range(train_times):
        th.set_grad_enabled(True)
        net.train()
        inp, lab = seq_data.sample_for_train(batch_size=batch_size, seq_len=seq_len, device=device)
        out, _ = net(inp)
        obj = net.get_obj_value(criterion=criterion, out=out, lab=lab, wup_dim=wup_dim)
        _update_network(optimizer, obj.mean(), clip_grad_norm)

        evaluator.update_obj_train(obj=obj)

        if (step_idx % valid_gap == 0) or (step_idx == train_times - 1):
            th.set_grad_enabled(False)
            evaluator.update_obj_train(obj=None)

            '''update_obj_valid'''
            net.eval()
            valid_objs = []
            for _ in range(valid_batches):
                inp, lab = seq_data.sample_for_train(batch_size=batch_size, seq_len=seq_len, device=device)
                out, _ = net(inp)
                
                seq_len_ = min(out.shape[0], lab.shape[0])
                valid_objs.append(criterion(out[wup_dim:seq_len_], lab[wup_dim:seq_len_]))

            if valid_objs:
                # Concatenate all validation objectives before averaging
                valid_obj = th.cat(valid_objs, dim=0)
                evaluator.update_obj_valid(valid_obj)
            else:
                print("| Warning: Empty validation batch")
                evaluator.update_obj_valid(th.zeros(0))  # Handle empty case properly

            del inp, lab, out, valid_objs
            evaluator.update_obj_valid(obj=None)

            evaluator.log_print(step_idx=step_idx)
            evaluator.draw_train_valid_loss_curve(gpu_id=gpu_id)
            # validator.draw_roc_curve_and_accuracy_curve(gpu_id=gpu_id, step_idx=0)

            if evaluator.patience > num_patience:
                break
            if evaluator.patience == 0:
                best_valid_loss = evaluator.best_valid_loss
                th.save(net.state_dict(), f'{out_dir}/net_{step_idx:06}_{best_valid_loss:06.3f}.pth')
                # validator.validate_save(f'{out_dir}_result.csv')

    predict_net_path = args.predict_net_path
    th.save(net.state_dict(), predict_net_path)
    print(f'| save network in {predict_net_path}')

    predict_ary = np.empty_like(seq_data.valid_label_seq.cpu().numpy())
    hid: Optional[TEN] = None

    print(f"| valid_seq_len {seq_data.valid_seq_len}  valid_times {seq_data.valid_seq_len // seq_len}")
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1].to(device)
        out, hid = net.forward(inp[:, None, :], hid)
        if hid is not None:
            predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)
    predict_ary_path = args.predict_ary_path
    np.save(predict_ary_path, predict_ary)
    print(f'| save predict in {predict_ary_path}')


def valid_model(gpu_id: int):
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    th.set_grad_enabled(False)

    '''data'''
    args = ConfigData()
    
    # 检查是否有mamba模型结果
    mamba_available = check_mamba_availability()
    mamba_paths = get_mamba_model_paths(args)
    
    if mamba_available and os.path.exists(mamba_paths['predict_path']):
        print("| Using existing Mamba prediction results")
        return
    elif mamba_available and os.path.exists(mamba_paths['net_path']):
        print("| Loading Mamba model for validation")
        valid_mamba_model(gpu_id)
        return
    
    print("| Using RNN model for validation")
    
    '''Config for training'''
    # 修改这里，使用与train_model相同的参数
    mid_dim = 128  # 从256改为128
    num_layers = 4  # 从8改为4

    seq_data = SeqData(args=args, train_ratio=0.0)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim

    predict_net_path = args.predict_net_path
    predict_ary_path = args.predict_ary_path

    '''Model'''
    from seq_net import RnnRegNet
    net = RnnRegNet(inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers).to(device)
    net.load_state_dict(th.load(predict_net_path, map_location=lambda storage, loc: storage))

    predict_ary = np.empty_like(seq_data.valid_label_seq.cpu().numpy() if seq_data.valid_label_seq.is_cuda else seq_data.valid_label_seq.numpy())
    hid: Optional[TEN] = None

    seq_len = 2 ** 9
    print(f"| valid_seq_len {seq_data.valid_seq_len}  valid_times {seq_data.valid_seq_len // seq_len}")
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1].to(device)
        out, hid = net.forward(inp[:, None, :], hid)
        if hid is not None:
            predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)
    np.save(predict_ary_path, predict_ary)
    print(f'| save predict in {predict_ary_path}')


def valid_mamba_model(gpu_id: int):
    """验证Mamba模型"""
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    th.set_grad_enabled(False)

    '''Config'''
    mid_dim = 128
    num_layers = 4

    '''data'''
    args = ConfigData()
    seq_data = SeqData(args=args, train_ratio=0.0)
    input_dim = seq_data.input_dim
    label_dim = seq_data.label_dim
    
    # Move data tensors to device to avoid CUDA errors
    seq_data.valid_input_seq = seq_data.valid_input_seq.to(device)
    seq_data.valid_label_seq = seq_data.valid_label_seq.to(device)
    
    mamba_paths = get_mamba_model_paths(args)

    '''Model'''
    from seq_net import MambaTCNRegNet
    net = MambaTCNRegNet(inp_dim=input_dim, mid_dim=mid_dim, out_dim=label_dim, num_layers=num_layers).to(device)
    net.load_state_dict(th.load(mamba_paths['net_path'], map_location=lambda storage, loc: storage))

    predict_ary = np.empty_like(seq_data.valid_label_seq.cpu().numpy())
    hid: Optional[TEN] = None

    seq_len = 2 ** 9
    print(f"| mamba valid_seq_len {seq_data.valid_seq_len}  valid_times {seq_data.valid_seq_len // seq_len}")
    for seq_i0 in range(0, seq_data.valid_seq_len, seq_len):
        seq_i1 = seq_i0 + seq_len
        inp = seq_data.valid_input_seq[seq_i0:seq_i1]
        out, hid = net.forward(inp[:, None, :], hid)
        if out is not None:
            predict_ary[seq_i0:seq_i1] = out.data.cpu().numpy().squeeze(1)
    
    np.save(mamba_paths['predict_path'], predict_ary)
    print(f'| save mamba predict in {mamba_paths["predict_path"]}')


if __name__ == '__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else -1  # Get GPU_ID from command line parameters
    convert_btc_csv_to_btc_npy()  # Data preprocessing, using market information and code to generate weak factor Alpha101
    train_model(gpu_id=GPU_ID)  # Using weak factor Alpha101 to train recurrent network RNN ​​(LSTM+GRU + Regression)
    valid_model(gpu_id=GPU_ID)  # Generate prediction results using the trained recurrent network and save them to the directory specified by ConfigData
