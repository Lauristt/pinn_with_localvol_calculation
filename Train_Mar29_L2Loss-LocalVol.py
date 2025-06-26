import numpy as np
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from datetime import datetime
import logging
import torch.distributions as dist # need use pipe
logging.basicConfig(level=logging.DEBUG)
from torch.utils.data import random_split
from torch.utils.checkpoint import checkpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import solver
from src.Solver import RBFLayer, DupireVolatilityModel, CrankNicolsonSolver, VolatilityPrecomputer

pd.set_option('display.max_columns', 30)

def bsm_price(S, K, T, r, sigma, option_type='call'):
    """计算BSM期权价格"""
    T = torch.clamp(T, min=1e-6)  # 防止除零错误

    d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
    d2 = d1 - sigma * torch.sqrt(T)
    normal = dist.Normal(0, 1)

    if option_type == 'call':
        price = S * normal.cdf(d1) - K * torch.exp(-r * T) * normal.cdf(d2)
    else:
        price = K * torch.exp(-r * T) * normal.cdf(-d2) - S * normal.cdf(-d1)
    return price

def get_working_directory():
    """获取当前工作目录"""
    return Path.cwd()

def create_folder(folder_name: str) -> Path:
    """在当前工作目录下创建文件夹（若不存在）"""
    folder_path = Path.cwd() / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)  # 确保递归创建，并且如果存在不会报错
    return folder_path

def check_file_exists(filename: str, extension: str) -> bool:
    """检查当前工作目录下是否存在指定文件"""
    filepath = os.path.join(os.getcwd(), f"{filename}.{extension}")
    return os.path.isfile(filepath)

def preprocess(dataset_path, underlying_asset_data, save_path):
    """
    数据预处理函数，包含自动缓存管理和严格数据校验
    """
    max_retry = 3  # 最大重试次数

    # 缓存验证函数
    def _validate_cache(cache_path):
        try:
            df = pd.read_csv(cache_path, parse_dates=['exdate', 'date'])

            # 关键字段校验
            required_columns = ['Time_to_expiration', 'forward_price',
                                'strike_price', 'impl_volatility']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"缓存文件缺少关键列: {missing}")

            # 空值校验（特别是波动率）
            null_report = df[required_columns].isnull().sum()
            if null_report.any():
                raise ValueError(f"缓存包含空值:\n{null_report.to_string()}")

            # 数值范围校验
            if (df['Time_to_expiration'] <= 0).any():
                raise ValueError("存在非正时间到期值")
            if (df['forward_price'] <= 0).any():
                raise ValueError("存在非正远期价格")
            if (df['impl_volatility'] <= 0).any():
                raise ValueError("存在非正波动率")

            return df

        except Exception as e:
            print(f"缓存验证失败: {str(e)}")

            # 重命名问题文件
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            invalid_path = cache_path.parent / f"invalid_cache_{timestamp}{cache_path.suffix}"
            try:
                cache_path.rename(invalid_path)
                print(f"问题缓存已归档至: {invalid_path}")
            except Exception as rename_err:
                print(f"缓存重命名失败: {str(rename_err)}")

            return None

    # 尝试加载有效缓存
    cache_path = Path(save_path)
    for attempt in range(1, max_retry + 1):
        if cache_path.exists():
            print(f"尝试加载缓存 (第{attempt}次)...")
            valid_df = _validate_cache(cache_path)
            if valid_df is not None:
                print("有效缓存加载成功")
                ref_df = pd.read_excel(underlying_asset_data)
                return valid_df, ref_df
        else:
            break

    # 无有效缓存时重新生成数据
    print("执行完整数据预处理流程...")

    # 加载原始数据并立即过滤波动率
    print(f"加载原始数据: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 关键过滤步骤：保留有效波动率数据
    initial_count = len(df)
    df = df[df['impl_volatility'].notna()]
    filtered_count = initial_count - len(df)
    print(f"过滤无效波动率数据: {filtered_count} 条")
    print(f"剩余有效数据量: {len(df)} 条")

    # 清理冗余列
    redundant_cols = ['suffix']
    for col in ['cp_flag', 'cfadj']:
        if col in df.columns and df[col].nunique() == 1:
            redundant_cols.append(col)
    df.drop(columns=redundant_cols, inplace=True, errors='ignore')
    print(f"已删除冗余列: {redundant_cols}")

    # 特征工程
    df['LinearApproximation'] = (df['best_bid'] + df['best_offer']) / 2
    df['exdate'] = pd.to_datetime(df['exdate'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 计算时间到期（精确到年）
    df['Time_to_expiration'] = (df['exdate'] - df['date']).dt.total_seconds() / (365 * 24 * 3600)
    df['strike_price'] = df['strike_price'] / 1000  # 标准化执行价格

    # 合并标的资产数据
    print("关联标的资产价格...")
    ref_df = pd.read_excel(underlying_asset_data)
    ref_df['时间'] = pd.to_datetime(ref_df['时间'])

    # 使用内连接确保价格匹配
    merged_df = df.merge(
        ref_df[['时间', '收盘价(元)']].rename(columns={'时间': 'date', '收盘价(元)': 'forward_price'}),
        on='date',
        how='inner'
    )

    # 检查合并结果
    if len(merged_df) == 0:
        raise ValueError("合并后数据为空，请检查日期匹配情况")
    print(f"合并后有效数据量: {len(merged_df)} 条")

    # 最终校验
    required_columns = ['Time_to_expiration', 'forward_price',
                        'strike_price', 'impl_volatility']
    validation = merged_df[required_columns]

    # 空值检查
    null_report = validation.isnull().sum()
    if null_report.any():
        raise ValueError(f"最终数据包含空值:\n{null_report.to_string()}")

    # 数值合理性检查
    if (validation['Time_to_expiration'] <= 0).any():
        raise ValueError("时间到期值必须为正数")
    if (validation['forward_price'] <= 0).any():
        raise ValueError("远期价格必须为正数")
    if (validation['impl_volatility'] <= 0).any():
        raise ValueError("波动率必须为正数")

    # 保存结果
    merged_df.to_csv(cache_path, index=False)
    print(f"预处理数据已保存至: {cache_path}")

    return merged_df, ref_df

def find_price(date, ref_df) -> float:
    price = ref_df.loc[ref_df['时间'] == date, '收盘价(元)']
    return price.iloc[0] if not price.empty else None

def find_volatility(df, exdate, strike_price) -> float: # find the mean of the volatility under the condition
    filtered_df = df[(df['exdate'] == exdate) & (df['strike_price'] == strike_price)]
    vola = filtered_df['impl_volatility'].mean()
    return vola

def prepare_data(df: pd.DataFrame, ref_df: pd.DataFrame, output_path: str) -> tuple:
    """
    数据准备函数，生成训练所需的张量数据
    """
    # 缓存管理函数
    def _handle_tensor_cache(cache_path): #管理必要tensor
        try:
            data = torch.load(cache_path)
            required_keys = ['X_real', 'u_real', 'X_exp', 'u_exp']

            # 基础校验
            for key in required_keys:
                if key not in data:
                    raise KeyError(f"缺少必要键: {key}")
                tensor = data[key]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"{key} 应为张量")
                if tensor.dtype != torch.float32:
                    raise TypeError(f"{key} 应为float32类型")
                if torch.isnan(tensor).any():
                    raise ValueError(f"{key} 包含NaN值")

            # 形状一致性校验
            n_samples = len(df)
            if data['X_real'].shape[0] != n_samples:
                raise ValueError(f"样本数不匹配: X_real={data['X_real'].shape[0]} vs df={n_samples}")

            return data

        except Exception as e:
            print(f"张量缓存校验失败: {str(e)}")

            # 归档问题文件
            if cache_path.exists():
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                invalid_path = cache_path.parent / f"invalid_tensors_{timestamp}.pt"
                try:
                    cache_path.rename(invalid_path)
                    print(f"问题张量已归档至: {invalid_path}")
                except Exception as rename_err:
                    print(f"张量归档失败: {str(rename_err)}")

            return None
    # 尝试加载有效张量缓存
    cache_path = Path(output_path)
    if cache_path.exists():
        print("检测到张量缓存，开始验证...")
        cache_data = _handle_tensor_cache(cache_path)
        if cache_data is not None:
            print("有效张量缓存已加载")
            return (cache_data['X_real'], cache_data['u_real'],
                    cache_data['X_exp'], cache_data['u_exp'])

    # 无有效缓存时生成新数据
    print("生成新的张量数据...")

    # 创建基础张量
    X_real = df[['Time_to_expiration', 'forward_price',
                 'strike_price', 'impl_volatility']].values
    X_real = torch.tensor(X_real, dtype=torch.float32)
    u_real = torch.tensor(df['LinearApproximation'].values, dtype=torch.float32).view(-1, 1)

    # 并行处理边界条件
    print("处理边界条件，使用线程池加速...")
    X_exp = np.zeros((len(df), 4))
    u_exp = []

    # 波动率缓存字典
    vol_cache = {}
    def process_row(i):
        row = df.iloc[i]
        exdate = row['exdate']
        strike = row['strike_price']
        fwd = row['forward_price']
        T = row['Time_to_expiration']
        r = 0.0525


        # 获取波动率（带缓存）
        cache_key = (exdate, strike)
        if cache_key not in vol_cache: # vol_cache = {(exdate1, strike1): vol1, (exdate2, strike2): vol2 }
            mask = (df['exdate'] == exdate) & (df['strike_price'] == strike)
            vol_cache[cache_key] = df.loc[mask, 'impl_volatility'].mean() # vol_cache -> float

        vol = vol_cache[cache_key]
        boundary_time = 0.0 if T < 1e-4 else T

        X_exp[i] = [
            boundary_time,  # 动态时间
            fwd,
            strike,
            vol
        ]

        if T < 1e-4:  # 接近到期的情况
            u_exp_val = max(fwd - strike, 0) #if call_else_put else max(strike - fwd, 0)
        else:
            u_exp_val = bsm_price(torch.tensor(fwd), torch.tensor(strike),
                                  torch.tensor(T), torch.tensor(r),
                                  torch.tensor(vol),option_type='call').item()
        # bsm_price(S, K, T, r, sigma, option_type='put')
        # if abs(T) < 1e-9:
        #     u_exp_val = max(fwd - strike, 0) if call_else_put else max(strike - fwd, 0)
        return i, u_exp_val

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row, i) for i in range(len(df))]
        for future in tqdm(futures, desc="处理进度", ncols=80):
            i, val = future.result()
            u_exp.append(val)

    # 转换为张量
    X_exp = torch.tensor(X_exp, dtype=torch.float32)
    u_exp = torch.tensor(u_exp, dtype=torch.float32).view(-1, 1)

    # 最终校验
    if torch.isnan(X_exp).any() or torch.isnan(u_exp).any():
        raise ValueError("生成的张量包含NaN值")

    # 保存结果
    torch.save({
        'X_real': X_real,
        'u_real': u_real,
        'X_exp': X_exp,
        'u_exp': u_exp
    }, cache_path)
    print(f"张量数据已保存至: {cache_path}")

    return X_real, u_real, X_exp, u_exp

class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.net(x)

class LocalVolatilityPINN(nn.Module):
    def __init__(self, dupire_model, hidden_dim=256, num_blocks=3):
        super().__init__()
        # 显式断开循环引用
        self.dupire = dupire_model
        self.solver = dupire_model.solver if hasattr(dupire_model, 'solver') else None

        # 优化网络结构避免递归
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        ).requires_grad_(True)  # 显式设置梯度要求

        # 简化残差块实现
        self.resnet = self._build_resnet(hidden_dim, num_blocks)

        # 优化预测头
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(inplace=True),  # 使用inplace操作节省内存
            nn.Linear(128, 1)
        )

        # 内存友好的权重初始化
        self.apply(self._init_weights)

        # 确保所有参数在相同设备
        self._device = next(self.parameters()).device

    def _build_resnet(self, hidden_dim, num_blocks):
        """构建内存高效的残差网络"""
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            )
        return nn.Sequential(*blocks, nn.LayerNorm(hidden_dim))

    def _init_weights(self, module):
        """改进的权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, X):
        # 优化特征提取流程
        T = X[:, 0]
        S = X[:, 1]
        K = X[:, 2]

        # 分批处理波动率计算
        batch_size = 4096  # 根据显存调整
        sigma_batches = []
        for i in range(0, len(K), batch_size):
            K_batch = K[i:i + batch_size]
            T_batch = T[i:i + batch_size]
            with torch.no_grad():  # 禁用梯度以节省显存
                sigma_batches.append(self.dupire.get_sigma(K_batch, T_batch))

        sigma = torch.cat(sigma_batches)
        features = torch.stack([T, S / K, sigma], dim=1).to(self._device)

        # 优化计算流程
        x = self.input_proj(features)
        x = self.resnet(x)
        return self.price_head(x)


def train_step(model, X_real_batch, u_real_batch, X_exp_batch, u_exp_batch, r, lambda_params):
    model.train()
    u_pred = model(X_real_batch)

    S, K, T, sigma = X_real_batch[:, 1], X_real_batch[:, 2], X_real_batch[:, 0], X_real_batch[:, 3]
    # bsm_prices = bsm_price(S, K, T, r, sigma, option_type = 'call').unsqueeze(-1)  # 保持维度一致性
    print(f"S shape: {S.shape}, K shape: {K.shape}, T shape: {T.shape}")

    with torch.no_grad():
        cn_prices = model.dupire.solver.batch_solve(S, K, T)

    cn_prices = torch.tensor(cn_prices, device=u_pred.device).unsqueeze(-1)

    # 计算原始损失
    raw_data_loss = F.mse_loss(u_pred, u_real_batch)
    print(u_pred.shape, u_real_batch.shape)
    raw_bsm_loss = F.mse_loss(cn_prices, u_pred)  # 维度已对齐，原始数据有误
    print(cn_prices.shape, u_pred.shape)
    u_exp_pred = model(X_exp_batch)
    raw_exp_loss = F.mse_loss(u_exp_pred, u_exp_batch)
    print(u_exp_pred.shape, u_exp_batch.shape)

    # 首次迭代时初始化基准损失
    if not hasattr(model, 'init_losses'):
        model.init_losses = {
            'data': max(raw_data_loss.item(), 1e-8),
            'bsm': max(raw_bsm_loss.item(), 1e-8),
            'exp': max(raw_exp_loss.item(), 1e-8)
        }
        lambda_params['init'] = model.init_losses.copy()
        lambda_params['prev'] = model.init_losses.copy()

    # 计算当前损失与初始损失的比值
    current_losses = {
        'data': max(raw_data_loss.item(), 1e-8),
        'bsm': max(raw_bsm_loss.item(), 1e-8),
        'exp': max(raw_exp_loss.item(), 1e-8)
    }

    # 带数值稳定的平衡权重计算
    def compute_balanced_lambda(current, init_ref, m=3):
        # 使用初始损失作为分母基准
        ratios = [current[k] / max(init_ref[k], 1e-8) for k in current]
        # 限制指数输入范围
        ratios = [np.clip(r, -50, 50) for r in ratios]
        exp_ratios = torch.tensor([np.exp(r) for r in ratios])
        sum_exp = max(exp_ratios.sum(), 1e-8)  # 防止除零
        return [m * e / sum_exp for e in exp_ratios]

    # 基于初始损失的权重计算
    lambda_bal_init = compute_balanced_lambda(current_losses, lambda_params['init'])
    # 基于前一时段损失的权重计算
    lambda_bal_prev = compute_balanced_lambda(current_losses, lambda_params['prev'])

    # 历史权重更新
    new_lambda_hist = [
        lambda_params['rho'] * lambda_params['hist'][i] +
        (1 - lambda_params['rho']) * lambda_bal_init[i]
        for i in range(3)
    ]

    # 最终权重合成
    final_lambda = [
        lambda_params['alpha'] * new_lambda_hist[i] +
        (1 - lambda_params['alpha']) * lambda_bal_prev[i]
        for i in range(3)
    ]

    # 强制BSM损失权重下限
    final_lambda[1] = max(final_lambda[1], 0.1)

    # 更新参数记录
    lambda_params['prev'] = current_losses
    lambda_params['hist'] = new_lambda_hist

    # 计算加权总损失
    total_loss = (
            final_lambda[0] * raw_data_loss +
            final_lambda[1] * raw_bsm_loss +
            final_lambda[2] * raw_exp_loss
    )

    return {
        'total': total_loss,
        'data': final_lambda[0] * raw_data_loss,
        'bsm': final_lambda[1] * raw_bsm_loss,
        'exp': final_lambda[2] * raw_exp_loss,
        'raw_data': raw_data_loss,
        'raw_bsm': raw_bsm_loss,
        'raw_exp': raw_exp_loss,
    }, lambda_params


# 数据预处理部分
def preprocess_data(dataset_path, underlying_asset_data, updated_data_path,prepare_data_outcome_path):
    df, ref_df = preprocess(dataset_path, underlying_asset_data, updated_data_path)
    X_real, u_real, X_exp, u_exp = prepare_data(df, ref_df, prepare_data_outcome_path)

    full_dataset = TensorDataset(X_real, u_real, X_exp, u_exp)

    return full_dataset


def save_loss_records(loss_records, save_dir):
    """保存损失数据并生成可视化图表（包含原始/缩放损失的Epoch时间序列对比）

    Args:
        loss_records : 包含以下键的字典
            - epoch       : 每个epoch的缩放损失均值（list of dict）
            - epoch_raw_* : 每个epoch的原始损失均值（list）
            - batch       : 所有batch的索引（list）
            - total_loss  : 所有batch的总损失（list）
            - raw_*_loss  : 所有batch的原始损失（list）
        save_dir     : 结果保存路径
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # ====================================================================
    # 核心修改点：移除了与模型自带缩放参数相关的代码
    # ====================================================================

    # 保存BATCH级别的原始/缩放损失（原有功能）
    pd.DataFrame({
        'batch': loss_records['batch'],
        'scaled_total': loss_records['total_loss'],
        'scaled_data': loss_records['data_loss'],
        'scaled_bsm': loss_records['bsm_loss'],
        'scaled_exp': loss_records['exp_loss']
    }).to_csv(save_dir / "batch_loss_scaled.csv", index=False)

    pd.DataFrame({
        'batch': loss_records['batch'],
        'raw_data': loss_records['raw_data_loss'],
        'raw_bsm': loss_records['raw_bsm_loss'],
        'raw_exp': loss_records['raw_exp_loss']
    }).to_csv(save_dir / "batch_loss_raw.csv", index=False)

    # 新增EPOCH时间序列数据保存
    if 'epoch' in loss_records and len(loss_records['epoch']) > 0:
        epoch_df = pd.DataFrame({
            # 从每个epoch字典中提取数据
            'epoch': [e['epoch'] for e in loss_records['epoch']],
            'scaled_total': [e['total'] for e in loss_records['epoch']],
            'scaled_data': [e['data'] for e in loss_records['epoch']],
            'scaled_bsm': [e['bsm'] for e in loss_records['epoch']],
            'scaled_exp': [e['exp'] for e in loss_records['epoch']],

            # 直接读取预先计算好的原始损失均值
            'raw_data': loss_records.get('epoch_raw_data', []),
            'raw_bsm': loss_records.get('epoch_raw_bsm', []),
            'raw_exp': loss_records.get('epoch_raw_exp', [])
        })
        epoch_df.to_csv(save_dir / "epoch_loss_history.csv", index=False)

        # 生成Epoch趋势对比图
        plt.figure(figsize=(12, 6))
        plt.plot(epoch_df['epoch'], epoch_df['scaled_total'], 'b-', label='Scaled Total Loss')
        plt.plot(epoch_df['epoch'], epoch_df['raw_data'], 'r--', label='Raw Data Loss')
        plt.plot(epoch_df['epoch'], epoch_df['raw_bsm'], 'g--', label='Raw BSM Loss')
        plt.plot(epoch_df['epoch'], epoch_df['raw_exp'], 'm--', label='Raw Expiry Loss')
        plt.yscale('log')
        plt.title("Epoch-wise Loss Trends (Raw vs Custom Weighted)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.savefig(save_dir / "epoch_loss_trends.png")
        plt.close()

    # 保留修改后的损失分析图（移除了缩放因子可视化）
    plt.figure(figsize=(12, 6))
    plt.plot(loss_records['batch'], loss_records['total_loss'], label='Total Loss')
    plt.yscale('log')
    plt.title("Batch-wise Training Dynamics")
    plt.xlabel("Batch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "training_dynamics.png")
    plt.close()



# 主函数
def main() -> None:
    # 初始化损失记录字典（添加缩放因子记录）
    loss_records = {
        'batch': [],
        'epoch': [],
        'total_loss': [],
        'data_loss': [],
        'bsm_loss': [],
        'exp_loss': [],
        'raw_data_loss': [],  # 新增原始损失记录
        'raw_bsm_loss': [],
        'raw_exp_loss': [],
        'data_pct': [],
        'bsm_pct': [],
        'exp_pct': [],
        'scales': [],
        'val_mae': [],
        'val_rmse': [],
        'val_relerr': []
    }

    # 创建输出目录
    plot_dir = create_folder("training_plots")
    r = 0.0525  # 无风险利率
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n=== 训练设备: {device} ===")
    print(f"GPU加速状态: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"剩余显存: {torch.cuda.mem_get_info()[0]/1024**3:.1f} GB\n")

    # 数据路径配置
    dataset_path = Path.cwd() / 'processed_option_data.csv'
    underlying_asset_data = Path.cwd() / '行情序列.xlsx'
    updated_data_path = Path.cwd() / 'StandardizedData_Feb26.csv'

    # 创建检查点和缓存目录
    checkpoint_dir = create_folder("checkpoints")
    cache_dir = create_folder("RunningCache")
    prepare_data_outcome_path = checkpoint_dir / "prepare_data.pt"
    pred_export_path = cache_dir / "full_predictions.npy"
    parameters_export_path = cache_dir / "final_model.pth"
    vol_surface_png_path = cache_dir / "vol_surface.png"
    checkpoint_model_dir = checkpoint_dir / "model_saved" / 'best_model.pth'

    # 数据预处理（保持原始量纲）
    print("Start Preprocessing Data...")
    try:
        full_dataset = preprocess_data(
            dataset_path,
            underlying_asset_data,
            updated_data_path,
            prepare_data_outcome_path
        )
        torch.save(full_dataset, cache_dir / "raw_dataset.pt")
        print(f"Raw dataset saved to {cache_dir}")
    except Exception as e:
        print(f"Data preprocessing failed: {str(e)}")
        return

    # 数据集分割（添加类型检查）
    dataset_size = len(full_dataset)
    test_size = int(0.05 * dataset_size)
    train_size = dataset_size - test_size

    if not isinstance(full_dataset, TensorDataset):
        print("Invalid dataset type")
        return

    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 数据加载器配置（优化内存使用）
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True
    )

    # 模型初始化（添加参数检查）
    try:
        df_vol_cal = pd.read_csv(updated_data_path)
        strikes = torch.tensor(df_vol_cal['strike_price'].unique(), device=device)
        expiries = torch.tensor(df_vol_cal['Time_to_expiration'].unique(), device=device)
        option_data = {
            'forward_price': df_vol_cal['forward_price'].values.astype(np.float32),
            'strike_price': df_vol_cal['strike_price'].values.astype(np.float32),
            'Time_to_expiration': df_vol_cal['Time_to_expiration'].values.astype(np.float32),
            'impl_volatility': df_vol_cal['impl_volatility'].values.astype(np.float32)
        }
        # use dupire to price volatility surface
        dupire = DupireVolatilityModel(
            strikes=strikes,
            expiries=expiries,
            option_data=option_data,
            r=0.05,
            device=device
        )
        solver = CrankNicolsonSolver(dupire, r=0.0525, q=0, device=device)
        dupire.solver = solver  # 将求解器附加到Dupire模型
        model = LocalVolatilityPINN(dupire).to(device)
        # 配置优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


    except AttributeError as e:
        print(f"Model initialization error: {str(e)}")
        return

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-4,
        total_steps=100 * len(train_loader),
        pct_start=0.3
    )

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 训练状态跟踪
    best_loss = float('inf')
    early_stop_counter = 0
    train_loss_history = []
    test_loss_history = []
    lr_history = []

    # 初始化lambda参数（添加默认值）
    lambda_params = {
        'rho': 0.9,  # 增大历史权重占比
        'alpha': 0.9,  # 平衡长短期基准
        'init': None,  # 由代码自动初始化
        'prev': None,  # 由代码自动更新
        'hist': [5.0] * 3  # 初始化为均匀权重
    }

    # 训练循环（添加异常处理）
    try:
        for epoch in range(100):
            raw_data_sum, raw_bsm_sum, raw_exp_sum = 0.0, 0.0, 0.0
            model.train()
            epoch_data = {'total': 0.0, 'data': 0.0, 'bsm': 0.0, 'exp': 0.0}
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch:03d}')

            for batch_idx, batch in enumerate(progress_bar):

                # 数据加载（添加设备转换检查）
                try:
                    X_real_batch, u_real_batch, X_exp_batch, u_exp_batch = batch
                    X_real_batch = X_real_batch.to(device, non_blocking=True)
                    u_real_batch = u_real_batch.to(device, non_blocking=True)
                    X_exp_batch = X_exp_batch.to(device, non_blocking=True)
                    u_exp_batch = u_exp_batch.to(device, non_blocking=True)
                except RuntimeError as e:
                    print(f"Data loading error: {str(e)}")
                    continue

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss_dict, lambda_params = train_step(
                        model,
                        X_real_batch,
                        u_real_batch,
                        X_exp_batch,
                        u_exp_batch,
                        r,
                        lambda_params
                    )
                raw_data_sum += loss_dict['raw_data'].item()
                raw_bsm_sum += loss_dict['raw_bsm'].item()
                raw_exp_sum += loss_dict['raw_exp'].item()

                loss_records['raw_data_loss'].append(loss_dict['raw_data'].item())
                loss_records['raw_bsm_loss'].append(loss_dict['raw_bsm'].item())
                loss_records['raw_exp_loss'].append(loss_dict['raw_exp'].item())

                scaler.scale(loss_dict['total']).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)

                # 参数更新
                scaler.step(optimizer)
                scaler.update()


                # 计算加权百分比（添加数值稳定性处理）
                total_weighted = loss_dict['total'].item()
                data_weighted = loss_dict['data'].item()
                bsm_weighted = loss_dict['bsm'].item()
                exp_weighted = loss_dict['exp'].item()

                if total_weighted > 1e-8:
                    data_pct = data_weighted / total_weighted * 100 #这里可能还要调整
                    bsm_pct = bsm_weighted / total_weighted * 100
                    exp_pct = exp_weighted / total_weighted * 100
                else:
                    data_pct = bsm_pct = exp_pct = 0.0

                # 更新进度条显示
                progress_bar.set_postfix({
                    'Data%': f"{data_pct:.1f}%",
                    'BSM%': f"{bsm_pct:.1f}%",
                    'Exp%': f"{exp_pct:.1f}%",
                })

                # 记录批次数据（添加类型转换）
                loss_records['batch'].append(batch_idx)
                loss_records['total_loss'].append(float(total_weighted))
                loss_records['data_loss'].append(float(data_weighted))
                loss_records['bsm_loss'].append(float(bsm_weighted))
                loss_records['exp_loss'].append(float(exp_weighted))
                loss_records['data_pct'].append(float(data_pct))
                loss_records['bsm_pct'].append(float(bsm_pct))
                loss_records['exp_pct'].append(float(exp_pct))

                # 累积epoch统计
                epoch_data['total'] += total_weighted
                epoch_data['data'] += data_weighted
                epoch_data['bsm'] += bsm_weighted
                epoch_data['exp'] += exp_weighted

            with torch.no_grad():
                new_sigma = dupire._init_sigma_grid()
                dupire.sigma_grid.data = 0.9 * dupire.sigma_grid + 0.1 * new_sigma

            n_batches = len(train_loader)
            loss_records.setdefault('epoch_raw_data', []).append(raw_data_sum / n_batches)
            loss_records.setdefault('epoch_raw_bsm', []).append(raw_bsm_sum / n_batches)
            loss_records.setdefault('epoch_raw_exp', []).append(raw_exp_sum / n_batches)

            # 更新学习率
            scheduler.step()

            # 记录epoch数据
            n_batches = len(train_loader)
            loss_records['epoch'].append({
                'epoch': epoch,
                'total': epoch_data['total'] / n_batches,
                'data': epoch_data['data'] / n_batches,
                'bsm': epoch_data['bsm'] / n_batches,
                'exp': epoch_data['exp'] / n_batches
            })

            # 验证步骤（添加模型状态保护）
            model.eval()
            test_loss = 0.0
            all_mae, all_rmse, all_relerr = [], [], []
            with torch.no_grad():
                for X_real_batch, u_real_batch, X_exp_batch, u_exp_batch in test_loader:
                    try:
                        X_real_batch = X_real_batch.to(device)
                        u_real_batch = u_real_batch.to(device)
                        X_exp_batch = X_exp_batch.to(device)
                        u_exp_batch = u_exp_batch.to(device)
                    except RuntimeError as e:
                        print(f"Validation data error: {str(e)}")
                        continue

                    # 边界条件验证
                    u_pred = model(X_exp_batch)
                    bc_loss = F.mse_loss(u_pred, u_exp_batch)# 为什么使用100
                    test_loss += bc_loss.item()

                    mae = torch.abs(u_pred - u_exp_batch).mean().item()
                    rmse = torch.sqrt(F.mse_loss(u_pred, u_exp_batch)).item()
                    relative_err = (torch.abs(u_pred - u_exp_batch) / (torch.abs(u_exp_batch) + 1e-8)).mean().item()

                    all_mae.append(mae)
                    all_rmse.append(rmse)
                    all_relerr.append(relative_err)


            # 早停机制
            avg_test_loss = test_loss / len(test_loader)
            loss_records['val_mae'].append(np.mean(all_mae))
            loss_records['val_rmse'].append(np.mean(all_rmse))
            loss_records['val_relerr'].append(np.mean(all_relerr) * 100)

            print(f"\nEpoch {epoch} 验证结果:")
            print(f"边界条件损失: {avg_test_loss:.4f}")
            print(f"MAE: {loss_records['val_mae'][-1]:.4f} | RMSE: {loss_records['val_rmse'][-1]:.4f}")
            print(f"相对误差: {loss_records['val_relerr'][-1]:.2f}%")
            save_loss_records(loss_records, cache_dir)

            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter >= 20:
                    print("Early stopping triggered")
                    break

            # 保存检查点
            torch.save(model.state_dict(), checkpoint_dir / f'epoch_{epoch}.pth')

            # 实时可视化（每5个epoch）
            if epoch % 5 == 0:
                plt.figure(figsize=(15, 5))

                # 损失曲线
                plt.subplot(1, 2, 1)
                plt.plot(loss_records['total_loss'], label='Total')
                plt.plot(loss_records['data_loss'], label='Data')
                plt.plot(loss_records['bsm_loss'], label='BSM')
                plt.plot(loss_records['exp_loss'], label='Expiry')
                plt.yscale('log')
                plt.legend()

                # 权重分布
                plt.subplot(1, 2, 2)
                plt.bar(['Data', 'BSM', 'Expiry'], [
                    np.mean(loss_records['data_pct']),
                    np.mean(loss_records['bsm_pct']),
                    np.mean(loss_records['exp_pct'])
                ])
                plt.ylim(0, 100)

                # 缩放因子趋势
                # plt.subplot(1, 3, 3)
                # scales = np.array(loss_records['scales'])
                # plt.plot(scales[:, 0], label='Data Scale')
                # plt.plot(scales[:, 1], label='BSM Scale')
                # plt.plot(scales[:, 2], label='Expiry Scale')
                # plt.legend()

                plt.tight_layout()
                plt.savefig(plot_dir / f'live_monitor_epoch_{epoch}.png')
                plt.close()

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return

    # 最终保存和输出
    try:
        dupire.plot_vol_surface(save_path=vol_surface_png_path, show_flag=False)  # check compatibility
        print("Vol Surface save success!")
        # save loss records
        save_loss_records(loss_records, cache_dir)

        # load the best model
        model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))

        # 生成预测
        model.eval()
        with torch.no_grad():
            all_preds = []
            for batch in DataLoader(full_dataset, batch_size=4096):
                X_real, _, _, _ = batch
                all_preds.append(model(X_real.to(device)).cpu())
            full_pred = torch.cat(all_preds).numpy()

        # 保存结果
        np.save(pred_export_path, full_pred)
        torch.save(model.state_dict(), parameters_export_path)

        print("Training completed successfully")
        print(f"Final predictions saved to {pred_export_path}")
        print(f"Model parameters saved to {parameters_export_path}")

    except Exception as e:
        print(f"Final save failed: {str(e)}")




    # def process_row(i) # deprecated:
    #     row = df.iloc[i]
    #     exdate = row['exdate']
    #     strike = row['strike_price']
    #     fwd = row['forward_price']
    #
    #
    #     # 获取波动率（带缓存）
    #     cache_key = (exdate, strike)
    #     if cache_key not in vol_cache:
    #         mask = (df['exdate'] == exdate) & (df['strike_price'] == strike)
    #         vol_cache[cache_key] = df.loc[mask, 'impl_volatility'].mean()
    #
    #     vol = vol_cache[cache_key]
    #
    #     X_exp[i] = [
    #         0.0,  # 时间到期（边界条件）
    #         row['forward_price'],
    #         strike,
    #         vol
    #     ]
    #
    #     u_exp_val = max(fwd - strike, 0.0)
    #     return i, u_exp_val
    #
    # with ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(process_row, i) for i in range(len(df))]
    #     for future in tqdm(futures, desc="处理进度", ncols=80):
    #         i, val = future.result()
    #         u_exp.append(val)
    #
    # # 转换为张量
    # X_exp = torch.tensor(X_exp, dtype=torch.float32)
    # u_exp = torch.tensor(u_exp, dtype=torch.float32).view(-1, 1)
    #
    # # 最终校验
    # if torch.isnan(X_exp).any() or torch.isnan(u_exp).any():
    #     raise ValueError("生成的张量包含NaN值")
    #
    # # 保存结果
    # torch.save({
    #     'X_real': X_real,
    #     'u_real': u_real,
    #     'X_exp': X_exp,
    #     'u_exp': u_exp
    # }, cache_path)
    # print(f"张量数据已保存至: {cache_path}")
    #
    # return X_real, u_real, X_exp, u_exp

if __name__ == "__main__":
    main()
