import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import weakref
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.checkpoint import checkpoint
from scipy.interpolate import RegularGridInterpolator


class VolatilityPrecomputer:
    def __init__(self, K_range=(3000, 5000), T_range=(0.01, 2.0), 
                 K_points=50, T_points=100):
        self.K_grid = np.linspace(K_range[0], K_range[1], K_points)
        self.T_grid = np.linspace(T_range[0], T_range[1], T_points)
        self.sigma_grid = None
        self.interp_fn = None
        self.solver = CrankNicolsonSolver()

    def precompute(self):
        """Precompute the Vol Surface"""
        print("Starting volatility precomputation...")
        self.sigma_grid = np.zeros((len(self.K_grid), len(self.T_grid)))
        
        for i, K in enumerate(self.K_grid):
            for j, T in enumerate(self.T_grid):
                # 调用完整FDM求解
                self.sigma_grid[i,j] = self.solver.solve(K=K, T=T)
            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{len(self.K_grid)} K points")
        
        # 创建插值器
        self.interp_fn = RegularGridInterpolator(
            (self.K_grid, self.T_grid), 
            self.sigma_grid,
            bounds_error=False,
            fill_value=None
        )
        print("Precomputation complete.")

    def save(self, path):
        """保存预计算结果"""
        np.savez(path, 
                 K_grid=self.K_grid,
                 T_grid=self.T_grid,
                 sigma_grid=self.sigma_grid)

    @classmethod
    def load(cls, path):
        """加载预计算结果"""
        data = np.load(path)
        instance = cls()
        instance.K_grid = data['K_grid']
        instance.T_grid = data['T_grid']
        instance.sigma_grid = data['sigma_grid']
        instance.interp_fn = RegularGridInterpolator(
            (instance.K_grid, instance.T_grid),
            instance.sigma_grid,
            bounds_error=False,
            fill_value=None
        )
        return instance

    def get_sigma(self, K, T):
        """获取插值波动率"""
        return self.interp_fn((K, T))



class RBFLayer(nn.Module):
    def __init__(self, centers, values, eps=1.0):
        super().__init__()
        # 使用register_buffer减少内存占用
        self.register_buffer('centers', centers)  # [N, 2]
        self.register_buffer('values', values)  # [N]
        self.eps = eps

    def forward(self, x):
        """内存优化版前向传播"""
        batch_size = 1024  # 减小批处理量
        results = []
        centers = self.centers.T  # [2, N] 转置提升内存局部性

        for i in range(0, x.size(0), batch_size):
            x_batch = x[i:i + batch_size].T  # [2, B]

            # 分维度计算减少峰值内存
            dist_sq = (x_batch[0].unsqueeze(1) - centers[0].unsqueeze(0)).pow(2)  # [B, N]
            dist_sq += (x_batch[1].unsqueeze(1) - centers[1].unsqueeze(0)).pow(2)

            # 低精度计算核函数
            with torch.cuda.amp.autocast():
                rbf = torch.exp(-(self.eps ** 2) * dist_sq)
                results.append(rbf.to(torch.float32) @ self.values)  # 转回float32保持精度

            # 显式释放中间变量
            del x_batch, dist_sq, rbf
            torch.cuda.empty_cache()

        return torch.cat(results)

class DupireVolatilityModel(nn.Module):
    def __init__(self, strikes, expiries, option_data, r, q=0, device='cuda'):
        super().__init__()
        self.device = device

        # 统一类型转换函数
        def _to_tensor(data):
            if isinstance(data, np.ndarray):
                return torch.tensor(data, device=self.device, dtype=torch.float32)
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            return torch.tensor([data], device=self.device, dtype=torch.float32).squeeze()

        # 转换核心参数
        self.strikes = _to_tensor(strikes).view(-1)
        self.expiries = _to_tensor(expiries).view(-1)
        self.r = _to_tensor(r)
        self.q = _to_tensor(q)

        # 转换option_data字典
        self.option_data = {
            'forward_price': _to_tensor(option_data['forward_price']),
            'impl_volatility': _to_tensor(option_data['impl_volatility']),
            'strike_price': _to_tensor(option_data['strike_price']),
            'Time_to_expiration': _to_tensor(option_data['Time_to_expiration'])
        }

        # 初始化基础参数
        self.S0 = self.option_data['forward_price'].mean()
        self.T_max = self.expiries.max()
        self.K_min = self.strikes.min()
        self.K_max = self.strikes.max()

        # 创建网格
        self.K_grid = torch.linspace(
            self.K_min, self.K_max, 60,
            device=device, dtype=torch.float32, requires_grad=False
        )

        self.T_grid = torch.linspace(
            self.expiries.min(), self.expiries.max(), 60,
            device=device, dtype=torch.float32, requires_grad=False
        )

        # 构建RBF插值器
        sample_idx = torch.randperm(len(self.option_data['forward_price']))[
                     :len(self.option_data['forward_price']) // 4]
        self.iv_interpolator = RBFLayer(
            centers=torch.stack([
                self.option_data['strike_price'][sample_idx],
                self.option_data['Time_to_expiration'][sample_idx]
            ], dim=1),
            values=self.option_data['impl_volatility'][sample_idx],
            eps=3.0
        )

        # 初始化可训练参数
        self.sigma_grid = nn.Parameter(
            self._init_sigma_grid(),
            requires_grad=True
        )
    def _init_sigma_grid(self):
        """优化分块策略"""
        iv = torch.zeros(60, 60, device=self.device)
        k_chunk = 20  # 减小分块尺寸
        t_chunk = 15

        with torch.no_grad():
            for i in range(0, 60, k_chunk):
                for j in range(0, 60, t_chunk):
                    # 动态调整实际分块大小
                    actual_k = min(k_chunk, 60 - i)
                    actual_t = min(t_chunk, 60 - j)

                    # 生成子网格
                    K_block = self.K_grid[i:i + actual_k]
                    T_block = self.T_grid[j:j + actual_t]

                    # 使用内存优化版meshgrid
                    K_mesh, T_mesh = torch.meshgrid(
                        K_block, T_block, indexing='ij'
                    )
                    points = torch.stack([K_mesh.ravel(), T_mesh.ravel()], dim=1)

                    # 分批次计算插值
                    sub_iv = []
                    sub_batch = 2048  # 子批次处理
                    for p in range(0, points.size(0), sub_batch):
                        sub_points = points[p:p + sub_batch]
                        sub_iv.append(self.iv_interpolator(sub_points))
                        del sub_points
                        torch.cuda.empty_cache()

                    iv_block = torch.cat(sub_iv).view(actual_k, actual_t)
                    iv[i:i + actual_k, j:j + actual_t] = iv_block

                    # 显式释放内存
                    del K_mesh, T_mesh, points, sub_iv, iv_block
                    torch.cuda.empty_cache()

        return self._compute_dupire_vol(iv, self.K_grid, self.T_grid)

    def _compute_dupire_vol(self, iv, K, T):
        sigma = torch.zeros(60, 60, device = self.device)
        chunk_size = 12  # 保持分块策略控制内存

        # 创建可微分视图（保持原始张量梯度）
        K = K.view(-1).requires_grad_(True)  # 移除detach()
        T = T.view(-1).requires_grad_(True)
        iv = iv.requires_grad_(True)

        for i in range(0, 60, chunk_size):
            for j in range(0, 60, chunk_size):
                # 直接使用原始张量的子视图（避免克隆）
                K_seg = K[i:i + chunk_size]
                T_seg = T[j:j + chunk_size]
                iv_seg = iv[i:i + chunk_size, j:j + chunk_size]

                # 生成可微分网格
                K_mesh, T_mesh = torch.meshgrid(
                    K_seg, T_seg, indexing='ij'
                )
                K_mesh.retain_grad()  # 显式保留梯度
                T_mesh.retain_grad()

                # 核心计算（保持完整计算图）
                log_term = torch.log(self.S0 / K_mesh)
                sqrt_T = torch.sqrt(T_mesh)
                d1 = (log_term + (self.r - self.q + 0.5 * iv_seg ** 2) * T_mesh) / (iv_seg * sqrt_T)
                d2 = d1 - iv_seg * sqrt_T
                C = self.S0 * torch.exp(-self.q * T_mesh) * torch.erf(d1 / np.sqrt(2)) \
                    - K_mesh * torch.exp(-self.r * T_mesh) * torch.erf(d2 / np.sqrt(2))

                # 梯度计算
                dC_dT = torch.autograd.grad(
                    C.sum(), T_mesh,
                    create_graph=True,
                    retain_graph=True
                )[0]

                dC_dK = torch.autograd.grad(
                    C.sum(), K_mesh,
                    create_graph=True,
                    retain_graph=True
                )[0]

                # 二阶导数（保持计算图）
                d2C_dK2 = torch.autograd.grad(
                    dC_dK.sum(), K_mesh,
                    create_graph=True,
                    retain_graph=True
                )[0]

                # 计算结果
                numerator = dC_dT + (self.r - self.q) * K_mesh * dC_dK
                denominator = 0.5 * K_mesh.pow(2) * d2C_dK2 + 1e-8
                sigma_block = torch.sqrt(numerator / denominator)

                # 分离结果防止梯度回传到网格
                sigma[i:i + chunk_size, j:j + chunk_size] = sigma_block.detach()

                # 显式释放中间变量
                del K_mesh, T_mesh, dC_dT, dC_dK, d2C_dK2, sigma_block
                torch.cuda.empty_cache()

        return torch.clamp(sigma, 0.05, 2.0)


    def plot_vol_surface(self, save_path='vol_surface.png', show_flag=True):
        """绘制3D波动率曲面"""
        # 转换为numpy数组
        K = self.K_grid.cpu().numpy()
        T = self.T_grid.cpu().numpy()
        sigma = self.sigma_grid.detach().cpu().numpy()

        # 创建网格
        K_mesh, T_mesh = np.meshgrid(K, T, indexing='ij')

        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K_mesh, T_mesh, sigma, cmap='viridis', rstride=10, cstride=10)

        # 添加标签和颜色条
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Volatility')
        ax.set_title('Local Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # 保存和显示
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_flag:
            plt.show()
        plt.close()

    def get_sigma(self, K, T):
        """双线性插值获取波动率"""
        K = K.to(self.device)
        T = T.to(self.device)

        # 寻找插值索引
        t_idx = torch.clamp(
            torch.searchsorted(self.T_grid, T),
            1, len(self.T_grid)-1
        )
        k_idx = torch.clamp(
            torch.searchsorted(self.K_grid, K),
            1, len(self.K_grid)-1
        )

        # 计算插值权重
        t_ratio = (T - self.T_grid[t_idx-1]) / (self.T_grid[t_idx] - self.T_grid[t_idx-1] + 1e-8)
        k_ratio = (K - self.K_grid[k_idx-1]) / (self.K_grid[k_idx] - self.K_grid[k_idx-1] + 1e-8)

        # 双线性插值
        return (
            (1 - t_ratio)*(1 - k_ratio)*self.sigma_grid[t_idx-1, k_idx-1] +
            t_ratio*(1 - k_ratio)*self.sigma_grid[t_idx, k_idx-1] +
            (1 - t_ratio)*k_ratio*self.sigma_grid[t_idx-1, k_idx] +
            t_ratio*k_ratio*self.sigma_grid[t_idx, k_idx]
        )

    def save_vol_surface(self, filename='Vol_surface_data.pt'):
        """保存波动率曲面数据"""
        torch.save({
            'K_grid': self.K_grid,
            'T_grid': self.T_grid,
            'sigma_grid': self.sigma_grid.data
        }, filename)


class CrankNicolsonSolver(nn.Module):
    def __init__(self, dupire_model, r, q, device='cuda'):
        super().__init__()
        self.dupire = weakref.ref(dupire_model)
        self.r = r
        self.q = q
        self.device = device

    def solve(self, S0, K, T, M=100, N=100):
        dupire = self.dupire()
        dt = T / N
        batch_size = S0.shape[0] if S0.dim() > 0 else 1
        S0 = S0.view(-1, 1)  # 确保二维张量 [batch, 1]

        # 动态调整网格范围（保持批量维度）
        Smax = 3 * torch.maximum(S0.squeeze(), K).unsqueeze(-1)  # [batch, 1]
        dS = Smax / M  # [batch, 1]

        # 生成批量网格（优化向量化）
        S = torch.cat([torch.linspace(0, Smax[i].item(), M + 1, device=self.device).unsqueeze(0)
                       for i in range(batch_size)], dim=0)  # [batch, M+1]
        t = torch.linspace(0, T, N + 1, device=self.device)

        # 初始化价值矩阵（添加批量维度）
        V = torch.zeros(batch_size, M + 1, N + 1, device=self.device)
        V[:,-1] = torch.maximum(S - K.view(-1,1), torch.tensor(0.0, device=self.device))

        # 批量计算局部波动率
        # S_mesh, t_mesh = torch.meshgrid(S, t, indexing='ij')
        # sigma = dupire.get_sigma(S_mesh, t_mesh)  # 已优化为直接处理网格输入

        # 预计算系数矩阵（向量化运算）
        S_exp = S.unsqueeze(-1).expand(-1, -1, N + 1)  # [batch, M+1, N+1]
        t_exp = t.expand(batch_size, M + 1, N + 1)  # [batch, M+1, N+1]
        sigma = dupire.get_sigma(S_exp, t_exp)  # 用户自定义波动率函数

        # 向量化系数计算（保持维度对齐）
        with torch.no_grad():
            S_sq = S_exp[:, 1:-1, :-1] ** 2  # 内部节点 [batch, M-1, N]
            dS_exp = dS.view(-1, 1, 1)  # [batch, 1, 1]

            a = 0.25 * dt * (sigma[:, 1:-1, :-1] ** 2 * S_sq / dS_exp ** 2 -
                             (self.r - self.q) * S_exp[:, 1:-1, :-1] / dS_exp)
            b = -0.5 * dt * (sigma[:, 1:-1, :-1] ** 2 * S_sq / dS_exp ** 2 + self.r)
            c = 0.25 * dt * (sigma[:, 1:-1, :-1] ** 2 * S_sq / dS_exp ** 2 +
                             (self.r - self.q) * S_exp[:, 1:-1, :-1] / dS_exp)

        # 在solve方法的时间循环中，调整以下部分：

        for j in range(N - 1, -1, -1):
            # 构造三对角系统（修复索引错误）
            main_diag = 1 - b[:, :, j]  # [batch, M-1]
            lower_diag = -a[:, 1:, j]  # 下对角线 [batch, M-2]
            upper_diag = -c[:, :-1, j]  # 上对角线 [batch, M-2]

            # 构造右端项（修正索引）
            rhs = (1 + b[:, :, j]) * V[:, 1:-1, j + 1] + \
                  a[:, :, j] * V[:, 2:, j + 1] + \
                  c[:, :, j] * V[:, :-2, j + 1]

            # 边界条件处理（保持原逻辑，索引已修正）
            rhs[:, 0] += a[:, 0, j] * V[:, 0, j + 1]
            rhs[:, -1] += c[:, -1, j] * V[:, -1, j + 1]

            # 调用batch_tdma_solve
            V[:, 1:-1, j] = self.batch_tdma_solve(main_diag, lower_diag, upper_diag, rhs)

        dupire.save_vol_surface()
        return torch.nn.functional.interpolate(
            V[:, :, 0].unsqueeze(1),
            size=S0.size(0),
            mode='linear',
            align_corners=True
        ).squeeze()

    def batch_tdma_solve(self, main_diag, lower_diag, upper_diag, rhs):
        """
        批量Thomas算法求解三对角方程组
        输入维度：
            main_diag: [batch_size, n]     主对角线元素
            lower_diag: [batch_size, n-1]  下对角线元素
            upper_diag: [batch_size, n-1]  上对角线元素
            rhs: [batch_size, n]           右端项
        输出维度：
            x: [batch_size, n]             解向量
        """
        batch_size, n = main_diag.size()
        device = main_diag.device

        # 克隆输入防止原地修改
        a = lower_diag.clone()  # 下对角线 [batch, n-1]
        b = main_diag.clone()  # 主对角线 [batch, n]
        c = upper_diag.clone()  # 上对角线 [batch, n-1]
        d = rhs.clone()  # 右端项 [batch, n]

        # 前向消元阶段
        # 处理第一个元素
        c_first = c[:, 0] / b[:, 0]
        d_first = d[:, 0] / b[:, 0]
        c = torch.cat([c_first.unsqueeze(1), c[:, 1:]], dim=1)
        d = torch.cat([d_first.unsqueeze(1), d[:, 1:]], dim=1)

        # 递归更新后续元素
        for i in range(1, n - 1):
            denom = b[:, i] - a[:, i - 1] * c[:, i - 1]
            c[:, i] = c[:, i] / denom
            d[:, i] = (d[:, i] - a[:, i - 1] * d[:, i - 1]) / denom

        # 处理最后一个元素
        denom_last = b[:, -1] - a[:, -1] * c[:, -2]
        d[:, -1] = (d[:, -1] - a[:, -1] * d[:, -2]) / denom_last

        # 回代阶段
        x = torch.zeros(batch_size, n, device=device)
        x[:, -1] = d[:, -1]
        for i in range(n - 2, -1, -1):
            x[:, i] = d[:, i] - c[:, i] * x[:, i + 1]

        return x

    def batch_solve(self, S_tensor, K_tensor, T_tensor):
        """批量求解器，支持GPU加速"""
        # 确保输入为张量且在同一设备
        S_tensor = torch.as_tensor(S_tensor, device=self.device)
        K_tensor = torch.as_tensor(K_tensor, device=self.device)
        T_tensor = torch.as_tensor(T_tensor, device=self.device)

        # 向量化计算
        prices = []
        for S0, K, T in zip(S_tensor, K_tensor, T_tensor):
            prices.append(self.solve(S0, K, T))
        return torch.stack(prices)

    def tdma_solve(self, a, b, c, d):
        """Thomas算法求解三对角方程组（GPU加速版）"""
        n = a.size(0)
        c_star = torch.zeros(n, device=self.device)
        d_star = torch.zeros(n, device=self.device)

        c_star[0] = c[0] / a[0]
        d_star[0] = d[0] / a[0]

        for i in range(1, n):
            denom = a[i] - b[i-1] * c_star[i-1]
            c_star[i] = c[i] / denom
            d_star[i] = (d[i] - b[i-1] * d_star[i-1]) / denom

        x = torch.zeros(n, device=self.device)
        x[-1] = d_star[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i+1]

        return x

    def _interpolate_price(self, S, V, S0):
        """二次插值获取精确价格"""
        S = S.contiguous()
        idx = torch.searchsorted(S, S0)

        if idx == 0 or idx == len(S):
            return V[idx]
        x0, x1, x2 = S[idx-1], S[idx], S[idx+1]
        y0, y1, y2 = V[idx-1], V[idx], V[idx+1]
        return ((S0 - x1)*(S0 - x2))/((x0 - x1)*(x0 - x2)) * y0 + \
               ((S0 - x0)*(S0 - x2))/((x1 - x0)*(x1 - x2)) * y1 + \
               ((S0 - x0)*(S0 - x1))/((x2 - x0)*(x2 - x1)) * y2
        
#newdevelopment
class CrankNicolsonSolver_solver(nn.Module):
    def __init__(self, dupire_model, r, q, device='cuda'):
        super().__init__()
        self.dupire = weakref.ref(dupire_model)
        self.r = r
        self.q = q
        self.device = device

    def solve(self, S0, K, T, M=100, N=100):
        dupire = self.dupire()
        dt = T / N
        batch_size = S0.shape[0] if S0.dim() > 0 else 1
        S0 = S0.view(-1, 1)  # 确保二维张量 [batch, 1]

        # 动态调整网格范围（保持批量维度）
        Smax = 3 * torch.maximum(S0.squeeze(), K).unsqueeze(-1)  # [batch, 1]
        dS = Smax / M  # [batch, 1]

        # 生成批量网格（优化向量化）
        S = torch.cat([torch.linspace(0, Smax[i].item(), M + 1, device=self.device).unsqueeze(0)
                       for i in range(batch_size)], dim=0)  # [batch, M+1]
        t = torch.linspace(0, T, N + 1, device=self.device)

        # 初始化价值矩阵（添加批量维度）
        V = torch.zeros(batch_size, M + 1, N + 1, device=self.device)
        V[:,-1] = torch.maximum(S - K.view(-1,1), torch.tensor(0.0, device=self.device))

        # 批量计算局部波动率
        # S_mesh, t_mesh = torch.meshgrid(S, t, indexing='ij')
        # sigma = dupire.get_sigma(S_mesh, t_mesh)  # 已优化为直接处理网格输入

        # 预计算系数矩阵（向量化运算）
        S_exp = S.unsqueeze(-1).expand(-1, -1, N + 1)  # [batch, M+1, N+1]
        t_exp = t.expand(batch_size, M + 1, N + 1)  # [batch, M+1, N+1]
        sigma = dupire.get_sigma(S_exp, t_exp)  # 用户自定义波动率函数

        # 向量化系数计算（保持维度对齐）
        with torch.no_grad():
            S_sq = S_exp[:, 1:-1, :-1] ** 2  # 内部节点 [batch, M-1, N]
            dS_exp = dS.view(-1, 1, 1)  # [batch, 1, 1]

            a = 0.25 * dt * (sigma[:, 1:-1, :-1] ** 2 * S_sq / dS_exp ** 2 -
                             (self.r - self.q) * S_exp[:, 1:-1, :-1] / dS_exp)
            b = -0.5 * dt * (sigma[:, 1:-1, :-1] ** 2 * S_sq / dS_exp ** 2 + self.r)
            c = 0.25 * dt * (sigma[:, 1:-1, :-1] ** 2 * S_sq / dS_exp ** 2 +
                             (self.r - self.q) * S_exp[:, 1:-1, :-1] / dS_exp)

        # 在solve方法的时间循环中，调整以下部分：

        for j in range(N - 1, -1, -1):
            # 构造三对角系统（修复索引错误）
            main_diag = 1 - b[:, :, j]  # [batch, M-1]
            lower_diag = -a[:, 1:, j]  # 下对角线 [batch, M-2]
            upper_diag = -c[:, :-1, j]  # 上对角线 [batch, M-2]

            # 构造右端项（修正索引）
            rhs = (1 + b[:, :, j]) * V[:, 1:-1, j + 1] + \
                  a[:, :, j] * V[:, 2:, j + 1] + \
                  c[:, :, j] * V[:, :-2, j + 1]

            # 边界条件处理（保持原逻辑，索引已修正）
            rhs[:, 0] += a[:, 0, j] * V[:, 0, j + 1]
            rhs[:, -1] += c[:, -1, j] * V[:, -1, j + 1]

            # 调用batch_tdma_solve
            V[:, 1:-1, j] = self.batch_tdma_solve(main_diag, lower_diag, upper_diag, rhs)

        dupire.save_vol_surface()
        return torch.nn.functional.interpolate(
            V[:, :, 0].unsqueeze(1),
            size=S0.size(0),
            mode='linear',
            align_corners=True
        ).squeeze()

    def batch_tdma_solve(self, main_diag, lower_diag, upper_diag, rhs):
        """
        批量Thomas算法求解三对角方程组
        输入维度：
            main_diag: [batch_size, n]     主对角线元素
            lower_diag: [batch_size, n-1]  下对角线元素
            upper_diag: [batch_size, n-1]  上对角线元素
            rhs: [batch_size, n]           右端项
        输出维度：
            x: [batch_size, n]             解向量
        """
        batch_size, n = main_diag.size()
        device = main_diag.device

        # 克隆输入防止原地修改
        a = lower_diag.clone()  # 下对角线 [batch, n-1]
        b = main_diag.clone()  # 主对角线 [batch, n]
        c = upper_diag.clone()  # 上对角线 [batch, n-1]
        d = rhs.clone()  # 右端项 [batch, n]

        # 前向消元阶段
        # 处理第一个元素
        c_first = c[:, 0] / b[:, 0]
        d_first = d[:, 0] / b[:, 0]
        c = torch.cat([c_first.unsqueeze(1), c[:, 1:]], dim=1)
        d = torch.cat([d_first.unsqueeze(1), d[:, 1:]], dim=1)

        # 递归更新后续元素
        for i in range(1, n - 1):
            denom = b[:, i] - a[:, i - 1] * c[:, i - 1]
            c[:, i] = c[:, i] / denom
            d[:, i] = (d[:, i] - a[:, i - 1] * d[:, i - 1]) / denom

        # 处理最后一个元素
        denom_last = b[:, -1] - a[:, -1] * c[:, -2]
        d[:, -1] = (d[:, -1] - a[:, -1] * d[:, -2]) / denom_last

        # 回代阶段
        x = torch.zeros(batch_size, n, device=device)
        x[:, -1] = d[:, -1]
        for i in range(n - 2, -1, -1):
            x[:, i] = d[:, i] - c[:, i] * x[:, i + 1]

        return x

    def batch_solve(self, S_tensor, K_tensor, T_tensor):
        """批量求解器，支持GPU加速"""
        # 确保输入为张量且在同一设备
        S_tensor = torch.as_tensor(S_tensor, device=self.device)
        K_tensor = torch.as_tensor(K_tensor, device=self.device)
        T_tensor = torch.as_tensor(T_tensor, device=self.device)

        # 向量化计算
        prices = []
        for S0, K, T in zip(S_tensor, K_tensor, T_tensor):
            prices.append(self.solve(S0, K, T))
        return torch.stack(prices)

    def tdma_solve(self, a, b, c, d):
        """Thomas算法求解三对角方程组（GPU加速版）"""
        n = a.size(0)
        c_star = torch.zeros(n, device=self.device)
        d_star = torch.zeros(n, device=self.device)

        c_star[0] = c[0] / a[0]
        d_star[0] = d[0] / a[0]

        for i in range(1, n):
            denom = a[i] - b[i-1] * c_star[i-1]
            c_star[i] = c[i] / denom
            d_star[i] = (d[i] - b[i-1] * d_star[i-1]) / denom

        x = torch.zeros(n, device=self.device)
        x[-1] = d_star[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i+1]

        return x

    def _interpolate_price(self, S, V, S0):
        """二次插值获取精确价格"""
        S = S.contiguous()
        idx = torch.searchsorted(S, S0)

        if idx == 0 or idx == len(S):
            return V[idx]
        x0, x1, x2 = S[idx-1], S[idx], S[idx+1]
        y0, y1, y2 = V[idx-1], V[idx], V[idx+1]
        return ((S0 - x1)*(S0 - x2))/((x0 - x1)*(x0 - x2)) * y0 + \
               ((S0 - x0)*(S0 - x2))/((x1 - x0)*(x1 - x2)) * y1 + \
               ((S0 - x0)*(S0 - x1))/((x2 - x0)*(x2 - x1)) * y2


class RBFLayer(nn.Module):
    def __init__(self, centers, values, eps=1.0):
        super().__init__()
        # 使用register_buffer减少内存占用
        self.register_buffer('centers', centers)  # [N, 2]
        self.register_buffer('values', values)  # [N]
        self.eps = eps

    def forward(self, x):
        """优化后的内存高效实现"""
        # x: [M, 2]
        batch_size = 4096  # 根据显存调整批处理大小
        results = []

        # 分批处理输入数据
        for i in range(0, x.size(0), batch_size):
            x_batch = x[i:i + batch_size]

            # 优化距离计算方式
            # 展开计算 (x1 - c1)^2 + (x2 - c2)^2
            diff_k = x_batch[:, 0].unsqueeze(1) - self.centers[:, 0].unsqueeze(0)  # [B, N]
            diff_t = x_batch[:, 1].unsqueeze(1) - self.centers[:, 1].unsqueeze(0)  # [B, N]
            dist_sq = diff_k.pow(2) + diff_t.pow(2)  # [B, N]

            # 应用RBF核
            rbf = torch.exp(-(self.eps ** 2) * dist_sq)
            results.append(rbf @ self.values)

        return torch.cat(results)


class DupireVolatilityModel(nn.Module):
    def __init__(self, strikes, expiries, option_data, r, q=0, device='cuda'):
        super().__init__()
        self.device = device

        # 统一类型转换函数
        def _to_tensor(data):
            if isinstance(data, np.ndarray):
                return torch.tensor(data, device=self.device, dtype=torch.float32)
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            return torch.tensor([data], device=self.device, dtype=torch.float32).squeeze()

        # 转换核心参数
        self.strikes = _to_tensor(strikes).view(-1)
        self.expiries = _to_tensor(expiries).view(-1)
        self.r = _to_tensor(r)
        self.q = _to_tensor(q)

        # 转换option_data字典
        self.option_data = {
            'forward_price': _to_tensor(option_data['forward_price']),
            'impl_volatility': _to_tensor(option_data['impl_volatility']),
            'strike_price': _to_tensor(option_data['strike_price']),
            'Time_to_expiration': _to_tensor(option_data['Time_to_expiration'])
        }

        # 初始化基础参数
        self.S0 = self.option_data['forward_price'].mean()
        self.T_max = self.expiries.max()
        self.K_min = self.strikes.min()
        self.K_max = self.strikes.max()

        # 创建网格
        self.K_grid = torch.linspace(
            self.K_min, self.K_max, 60,
            device=device, dtype=torch.float32, requires_grad=False
        )

        self.T_grid = torch.linspace(
            self.expiries.min(), self.expiries.max(), 60,
            device=device, dtype=torch.float32, requires_grad=False
        )

        # 构建RBF插值器
        sample_idx = torch.randperm(len(self.option_data['forward_price']))[
                     :len(self.option_data['forward_price']) // 4]
        self.iv_interpolator = RBFLayer(
            centers=torch.stack([
                self.option_data['strike_price'][sample_idx],
                self.option_data['Time_to_expiration'][sample_idx]
            ], dim=1),
            values=self.option_data['impl_volatility'][sample_idx],
            eps=3.0
        )

        # 初始化可训练参数
        self.sigma_grid = nn.Parameter(
            self._init_sigma_grid(),
            requires_grad=True
        )

    def _init_sigma_grid(self):
        """严格维度控制的分块计算"""
        iv = torch.zeros(len(self.K_grid), len(self.T_grid), device=self.device)

        # 确保网格点数可被分块大小整除
        k_chunk = 60
        t_chunk = 60
        print(f"调试信息 -> K网格形状: {self.K_grid.shape}, T网格形状: {self.T_grid.shape}")
        assert len(self.T_grid) == 60, f"T网格长度应为60，实际得到：{len(self.T_grid)}，请检查expiries输入数据范围"

        with torch.no_grad():
            # K维度处理
            for i in range(0, len(self.K_grid), k_chunk):
                K_block = self.K_grid[i:i + k_chunk].squeeze()
                actual_k = K_block.size(0)

                # T维度处理
                for j in range(0, len(self.T_grid), t_chunk):
                    T_block = self.T_grid.squeeze()[j:j + t_chunk]
                    actual_t = T_block.size(0)

                    # 生成精确网格
                    K_mesh, T_mesh = torch.meshgrid(
                        K_block,
                        T_block,
                        indexing='ij'
                    )
                    points = torch.stack([
                        K_mesh.reshape(-1),
                        T_mesh.reshape(-1)
                    ], dim=1)

                    # 计算插值结果
                    iv_values = self.iv_interpolator(points)

                    # 严格形状验证
                    assert iv_values.size(0) == actual_k * actual_t, \
                        f"插值结果数量{iv_values.size(0)} != 预期{actual_k * actual_t}"
                    iv_block = iv_values.view(actual_k, actual_t)

                    # 维度精确匹配
                    target_slice = iv[i:i + actual_k, j:j + actual_t]
                    assert target_slice.shape == iv_block.shape, \
                        f"目标形状{target_slice.shape} != 数据形状{iv_block.shape}"

                    iv[i:i + actual_k, j:j + actual_t] = iv_block

                    # 显式内存清理
                    del K_mesh, T_mesh, points, iv_values, iv_block
                    torch.cuda.empty_cache()

        return self._compute_dupire_vol(iv, self.K_grid, self.T_grid)

    def _compute_dupire_vol(self, iv, K, T):
        K = K.detach().requires_grad_(True)
        T = T.detach().requires_grad_(True)
        iv = iv.detach().requires_grad_(True)

        torch.cuda.empty_cache()

        # 生成二维网格
        K_mesh, T_mesh = torch.meshgrid(K, T, indexing='ij')

        # 替换原有断言
        assert K_mesh.dim() == 2 and T_mesh.dim() == 2, f"实际维度 K:{K_mesh.dim()}, T:{T_mesh.dim()}"

        with torch.cuda.amp.autocast():
            # 使用生成的二维网格
            K = K_mesh.unsqueeze(-1)  # [50, 60, 1]
            T = T_mesh.unsqueeze(-1)  # [50, 60, 1]
            iv = iv.unsqueeze(-1)  # [50, 60, 1]

            log_term = torch.log(self.S0 / K)
            sqrt_T = torch.sqrt(T)

            d1 = (log_term + (self.r - self.q + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
            d2 = d1 - iv * sqrt_T

            C = self.S0 * torch.exp(-self.q * T_mesh) * torch.erf(d1 / np.sqrt(2)) - \
                K_mesh * torch.exp(-self.r * T_mesh) * torch.erf(d2 / np.sqrt(2))

            dC_dT = torch.autograd.grad(C.sum(), T, create_graph=True, retain_graph=True)[0]
            dC_dK = torch.autograd.grad(C.sum(), K, create_graph=True, retain_graph=True)[0]
            d2C_dK2 = torch.autograd.grad(dC_dK.sum(), K, create_graph=True, retain_graph=True)[0]

            del C, dC_dK
            torch.cuda.empty_cache()

            numerator = dC_dT + (self.r - self.q) * K * dC_dK + self.q * C
            denominator = 0.5 * K ** 2 * d2C_dK2
            sigma = torch.sqrt(numerator / (denominator + 1e-8))

        return torch.clamp(sigma.detach(), 0.05, 2.0)

    def plot_vol_surface(self, save_path='vol_surface.png', show_flag=True):
        """绘制3D波动率曲面"""
        # 转换为numpy数组
        K = self.K_grid.cpu().numpy()
        T = self.T_grid.cpu().numpy()
        sigma = self.sigma_grid.detach().cpu().numpy()

        # 创建网格
        K_mesh, T_mesh = np.meshgrid(K, T, indexing='ij')

        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(K_mesh, T_mesh, sigma, cmap='viridis', rstride=10, cstride=10)

        # 添加标签和颜色条
        ax.set_xlabel('Strike Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Volatility')
        ax.set_title('Local Volatility Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # 保存和显示
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_flag:
            plt.show()
        plt.close()

    def get_sigma(self, K, T):
        """双线性插值获取波动率"""
        K = K.to(self.device)
        T = T.to(self.device)

        # 寻找插值索引
        t_idx = torch.clamp(
            torch.searchsorted(self.T_grid, T),
            1, len(self.T_grid)-1
        )
        k_idx = torch.clamp(
            torch.searchsorted(self.K_grid, K),
            1, len(self.K_grid)-1
        )

        # 计算插值权重
        t_ratio = (T - self.T_grid[t_idx-1]) / (self.T_grid[t_idx] - self.T_grid[t_idx-1] + 1e-8)
        k_ratio = (K - self.K_grid[k_idx-1]) / (self.K_grid[k_idx] - self.K_grid[k_idx-1] + 1e-8)

        # 双线性插值
        return (
            (1 - t_ratio)*(1 - k_ratio)*self.sigma_grid[t_idx-1, k_idx-1] +
            t_ratio*(1 - k_ratio)*self.sigma_grid[t_idx, k_idx-1] +
            (1 - t_ratio)*k_ratio*self.sigma_grid[t_idx-1, k_idx] +
            t_ratio*k_ratio*self.sigma_grid[t_idx, k_idx]
        )

    def save_vol_surface(self, filename='Vol_surface_data.pt'):
        """保存波动率曲面数据"""
        torch.save({
            'K_grid': self.K_grid,
            'T_grid': self.T_grid,
            'sigma_grid': self.sigma_grid.data
        }, filename)



    def _compute_dupire_vol(self, iv, K, T):
        K = K.detach().requires_grad_(True)
        T = T.detach().requires_grad_(True)
        iv = iv.detach().requires_grad_(True)

        torch.cuda.empty_cache()

        # 生成二维网格
        K_mesh, T_mesh = torch.meshgrid(K, T, indexing='ij')

        # 替换原有断言
        assert K_mesh.dim() == 2 and T_mesh.dim() == 2, f"实际维度 K:{K_mesh.dim()}, T:{T_mesh.dim()}"

        with torch.cuda.amp.autocast():
            # 使用生成的二维网格
            K = K_mesh.unsqueeze(-1)  # [50, 60, 1]
            T = T_mesh.unsqueeze(-1)  # [50, 60, 1]
            iv = iv.unsqueeze(-1)  # [50, 60, 1]

            log_term = torch.log(self.S0 / K)
            sqrt_T = torch.sqrt(T)

            d1 = (log_term + (self.r - self.q + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
            d2 = d1 - iv * sqrt_T

            C = self.S0 * torch.exp(-self.q * T_mesh) * torch.erf(d1 / np.sqrt(2)) - \
                K_mesh * torch.exp(-self.r * T_mesh) * torch.erf(d2 / np.sqrt(2))

            dC_dT = torch.autograd.grad(C.sum(), T, create_graph=True, retain_graph=True)[0]
            dC_dK = torch.autograd.grad(C.sum(), K, create_graph=True, retain_graph=True)[0]
            d2C_dK2 = torch.autograd.grad(dC_dK.sum(), K, create_graph=True, retain_graph=True)[0]

            del C, dC_dK
            torch.cuda.empty_cache()

            numerator = dC_dT + (self.r - self.q) * K * dC_dK + self.q * C
            denominator = 0.5 * K ** 2 * d2C_dK2
            sigma = torch.sqrt(numerator / (denominator + 1e-8))

        return torch.clamp(sigma.detach(), 0.05, 2.0)