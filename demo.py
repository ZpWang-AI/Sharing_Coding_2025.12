import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置为黑体

class NormalizationDemo:
    """归一化方法演示类"""
    
    def __init__(self):
        self.eps = 1e-5
        
    def generate_input(self, format_type: str = 'cv',
                    batch_size: int = 2,
                    height: int = 4,
                    width: int = 4,
                    channels: int = 3,
                    seq_len: int = 8,
                    seed: int = 42,
                    low: float = -2.0,   # 新增：均匀分布下限
                    high: float = 2.0):  # 新增：均匀分布上限
        """生成测试输入数据（均匀分布版本）"""
        np.random.seed(seed)
        
        if format_type == 'cv':
            shape = (batch_size, height, width, channels)
            # 核心改动：使用 np.random.uniform 生成均匀分布数据
            x = np.random.uniform(low=low, high=high, size=shape).astype(np.float32)
            print(f"生成 CV 格式数据（均匀分布），形状: {x.shape}")
            print(f"分布范围: [{low}, {high}]")
            
        elif format_type == 'nlp':
            shape = (batch_size, seq_len, channels)
            # 核心改动：使用 np.random.uniform 生成均匀分布数据
            x = np.random.uniform(low=low, high=high, size=shape).astype(np.float32)
            print(f"生成 NLP 格式数据（均匀分布），形状: {x.shape}")
            print(f"分布范围: [{low}, {high}]")
        else:
            raise ValueError(f"未知格式: {format_type}")
    
        # 统计信息保持不变
        print(f"原始数据统计: mean={x.mean():.4f}, std={x.std():.4f}, "
            f"min={x.min():.4f}, max={x.max():.4f}")
        return x
    
    def batch_norm(self, x: np.ndarray, gamma: float = 1.0, beta: float = 0.0,
                  format_type: str = 'cv') -> np.ndarray:
        """BatchNorm 实现"""
        print("\n" + "="*50)
        print(f"BatchNorm (gamma={gamma}, beta={beta})")
        
        # 根据格式调整轴
        if format_type == 'cv':
            # CV: (N, H, W, C)，在轴 (0,1,2) 上计算统计量
            axes = (0, 1, 2)
            keepdims = True
        else:  # nlp
            # NLP: (N, Seq, C)，在轴 (0,1) 上计算统计量
            axes = (0, 1)
            keepdims = True
        
        # 计算统计量
        mean = np.mean(x, axis=axes, keepdims=keepdims)
        var = np.var(x, axis=axes, keepdims=keepdims)
        
        print(f"计算统计量的轴: {axes}")
        print(f"均值形状: {mean.shape}")
        print(f"方差形状: {var.shape}")
        
        # 标准化
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        y = gamma * x_hat + beta
        
        # 打印结果统计
        self._print_statistics(x, x_hat, y, "BatchNorm")
        
        return y, {"mean": mean, "var": var, "x_hat": x_hat}
    
    def layer_norm(self, x: np.ndarray, gamma: float = 1.0, beta: float = 0.0,
                  format_type: str = 'cv') -> np.ndarray:
        """LayerNorm 实现"""
        print("\n" + "="*50)
        print(f"LayerNorm (gamma={gamma}, beta={beta})")
        
        # LayerNorm: 在最后一个轴上计算统计量 (特征维度)
        axes = -1
        keepdims = True
        
        # 计算统计量
        mean = np.mean(x, axis=axes, keepdims=keepdims)
        var = np.var(x, axis=axes, keepdims=keepdims)
        
        print(f"计算统计量的轴: {axes}")
        print(f"均值形状: {mean.shape}")
        print(f"方差形状: {var.shape}")
        
        # 标准化
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        y = gamma * x_hat + beta
        
        # 打印结果统计
        self._print_statistics(x, x_hat, y, "LayerNorm")
        
        return y, {"mean": mean, "var": var, "x_hat": x_hat}
    
    def instance_norm(self, x: np.ndarray, gamma: float = 1.0, beta: float = 0.0,
                     format_type: str = 'cv') -> np.ndarray:
        """InstanceNorm 实现"""
        print("\n" + "="*50)
        print(f"InstanceNorm (gamma={gamma}, beta={beta})")
        
        if format_type == 'cv':
            # CV: (N, H, W, C)，对每个样本和每个通道计算
            # 在轴 (1,2) 上计算统计量
            axes = (1, 2)
            keepdims = True
        else:  # nlp
            # NLP: (N, Seq, C)，对每个样本和每个通道计算
            # 在轴 (1) 上计算统计量
            axes = (1)
            keepdims = True
        
        # 计算统计量
        mean = np.mean(x, axis=axes, keepdims=keepdims)
        var = np.var(x, axis=axes, keepdims=keepdims)
        
        print(f"计算统计量的轴: {axes}")
        print(f"均值形状: {mean.shape}")
        print(f"方差形状: {var.shape}")
        
        # 标准化
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        
        # 缩放和平移
        y = gamma * x_hat + beta
        
        # 打印结果统计
        self._print_statistics(x, x_hat, y, "InstanceNorm")
        
        return y, {"mean": mean, "var": var, "x_hat": x_hat}
    
    def group_norm(self, x: np.ndarray, gamma: float = 1.0, beta: float = 0.0,
                  num_groups: int = 2, format_type: str = 'cv') -> np.ndarray:
        """GroupNorm 实现"""
        print("\n" + "="*50)
        print(f"GroupNorm (gamma={gamma}, beta={beta}, groups={num_groups})")
        
        # 获取输入形状
        if format_type == 'cv':
            N, H, W, C = x.shape
        else:  # nlp
            N, Seq, C = x.shape
        
        # 检查分组数是否整除通道数
        if C % num_groups != 0:
            raise ValueError(f"通道数 {C} 不能被分组数 {num_groups} 整除")
        
        group_size = C // num_groups
        
        # 重塑为分组形式
        if format_type == 'cv':
            # 重塑为 (N, H, W, num_groups, group_size)
            x_reshaped = x.reshape(N, H, W, num_groups, group_size)
            axes = (1, 2, 4)  # 在空间维度和组内通道维度计算
        else:  # nlp
            # 重塑为 (N, Seq, num_groups, group_size)
            x_reshaped = x.reshape(N, Seq, num_groups, group_size)
            axes = (1, 3)  # 在序列维度和组内通道维度计算
        
        # 计算统计量
        mean = np.mean(x_reshaped, axis=axes, keepdims=True)
        var = np.var(x_reshaped, axis=axes, keepdims=True)
        
        print(f"计算统计量的轴: {axes}")
        print(f"输入重塑后形状: {x_reshaped.shape}")
        print(f"均值形状: {mean.shape}")
        print(f"方差形状: {var.shape}")
        
        # 标准化
        x_hat = (x_reshaped - mean) / np.sqrt(var + self.eps)
        
        # 恢复原始形状
        x_hat = x_hat.reshape(x.shape)
        
        # 缩放和平移
        y = gamma * x_hat + beta
        
        # 打印结果统计
        self._print_statistics(x, x_hat, y, "GroupNorm")
        
        return y, {"mean": mean, "var": var, "x_hat": x_hat}
    
    def rms_norm(self, x: np.ndarray, gamma: float = 1.0,
                format_type: str = 'cv') -> np.ndarray:
        """RMSNorm 实现"""
        print("\n" + "="*50)
        print(f"RMSNorm (gamma={gamma})")
        
        # RMSNorm: 在最后一个轴上计算均方根
        axes = -1
        keepdims = True
        
        # 计算均方根 (RMS)
        mean_square = np.mean(x**2, axis=axes, keepdims=keepdims)
        rms = np.sqrt(mean_square + self.eps)
        
        print(f"计算 RMS 的轴: {axes}")
        print(f"RMS 形状: {rms.shape}")
        
        # 标准化 (注意：这里不减均值)
        x_hat = x / rms
        
        # 缩放 (注意：RMSNorm 通常没有 beta 参数)
        y = gamma * x_hat
        
        # 打印结果统计
        self._print_statistics(x, x_hat, y, "RMSNorm")
        
        return y, {"rms": rms, "x_hat": x_hat}
    
    def _print_statistics(self, x: np.ndarray, x_hat: np.ndarray, 
                         y: np.ndarray, norm_type: str):
        """打印统计信息"""
        print(f"\n{norm_type} 统计:")
        print(f"原始数据 x: mean={x.mean():.6f}, std={x.std():.6f}")
        print(f"标准化后 x_hat: mean={x_hat.mean():.6f}, std={x_hat.std():.6f}")
        print(f"输出 y: mean={y.mean():.6f}, std={y.std():.6f}")
        
        # 打印各维度统计
        print(f"\n各维度统计:")
        if x.ndim == 4:  # CV 格式
            print("维度: [N, H, W, C]")
            for n in range(min(2, x.shape[0])):  # 显示前2个批次
                print(f"  批次 {n}: mean={x[n].mean():.4f}, std={x[n].std():.4f}")
        else:  # NLP 格式
            print("维度: [N, Seq, C]")
            for n in range(min(2, x.shape[0])):  # 显示前2个批次
                print(f"  批次 {n}: mean={x[n].mean():.4f}, std={x[n].std():.4f}")
    
    def visualize_results(self, original: np.ndarray, results: dict, 
                         format_type: str = 'cv'):
        """可视化归一化结果"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 展平数据用于直方图
        original_flat = original.flatten()
        
        # 绘制原始数据
        axes[0].hist(original_flat, bins=50, alpha=0.7, color='blue')
        axes[0].axvline(original_flat.mean(), color='red', linestyle='--', label=f'mean={original_flat.mean():.4f}')
        axes[0].set_title('原始数据分布')
        axes[0].set_xlabel('值')
        axes[0].set_ylabel('频率')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制各归一化结果
        for idx, (name, (y, _)) in enumerate(results.items(), 1):
            if idx < 6:  # 最多显示5个结果
                y_flat = y.flatten()
                axes[idx].hist(y_flat, bins=50, alpha=0.7, color='green')
                axes[idx].axvline(y_flat.mean(), color='red', linestyle='--', label=f'mean={y_flat.mean():.4f}')
                axes[idx].axvline(y_flat.mean() - y_flat.std(), color='orange', linestyle=':', label=f'±1 std')
                axes[idx].axvline(y_flat.mean() + y_flat.std(), color='orange', linestyle=':')
                axes[idx].set_title(f'{name} 分布')
                axes[idx].set_xlabel('值')
                axes[idx].set_ylabel('频率')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(results)+1, 6):
            axes[i].axis('off')
        
        plt.suptitle(f'归一化方法比较 ({format_type.upper()} 格式)', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def demo_all_norms(self, format_type: str = 'cv', seed: int = 42):
        """演示所有归一化方法"""
        print(f"\n{'='*60}")
        print(f"演示 {format_type.upper()} 格式的所有归一化方法")
        print(f"{'='*60}")
        
        # 生成输入数据
        if format_type == 'cv':
            x = self.generate_input(format_type='cv', batch_size=10, 
                                   height=20, width=30, channels=6, seed=seed)
        else:
            x = self.generate_input(format_type='nlp', batch_size=10, 
                                   seq_len=20, channels=30, seed=seed)
        
        # 应用各种归一化方法
        results = {}
        
        # BatchNorm
        y_bn, _ = self.batch_norm(x, gamma=1.0, beta=0.1, format_type=format_type)
        results['BatchNorm'] = (y_bn, None)
        
        # LayerNorm
        y_ln, _ = self.layer_norm(x, gamma=1.0, beta=0.1, format_type=format_type)
        results['LayerNorm'] = (y_ln, None)
        
        # InstanceNorm
        y_in, _ = self.instance_norm(x, gamma=1.0, beta=0.1, format_type=format_type)
        results['InstanceNorm'] = (y_in, None)
        
        # GroupNorm (2组)
        y_gn, _ = self.group_norm(x, gamma=1.0, beta=0.1, num_groups=2, format_type=format_type)
        results['GroupNorm'] = (y_gn, None)
        
        # RMSNorm
        y_rms, _ = self.rms_norm(x, gamma=1.0, format_type=format_type)
        results['RMSNorm'] = (y_rms, None)
        
        # 可视化结果
        self.visualize_results(x, results, format_type)
        
        return x, results

# 运行演示
def main():
    demo = NormalizationDemo()
    
    # 演示 CV 格式
    print("\n" + "="*80)
    print("CV 格式演示")
    print("="*80)
    x_cv, results_cv = demo.demo_all_norms(format_type='cv', seed=42)
    
    # 演示 NLP 格式
    print("\n" + "="*80)
    print("NLP 格式演示")
    print("="*80)
    x_nlp, results_nlp = demo.demo_all_norms(format_type='nlp', seed=42)
    
    # 对比不同 gamma/beta 值的效果
    print("\n" + "="*80)
    print("不同 gamma/beta 参数效果对比")
    print("="*80)
    
    # 使用 NLP 格式演示参数效果
    x = demo.generate_input(format_type='nlp', batch_size=1, seq_len=5, channels=1, seed=42)
    x = x.squeeze()  # 变成 (Seq,) 方便查看
    
    print(f"\n原始数据: {x}")
    print(f"形状: {x.shape}")
    
    # LayerNorm 不同参数
    for gamma, beta in [(1.0, 0.0), (2.0, 0.0), (1.0, 1.0), (0.5, -0.5)]:
        y, _ = demo.layer_norm(x.reshape(1, 5, 1), gamma=gamma, beta=beta, format_type='nlp')
        y = y.squeeze()
        print(f"\nLayerNorm(gamma={gamma}, beta={beta}):")
        print(f"  输出: {y}")
        print(f"  均值: {y.mean():.4f}, 标准差: {y.std():.4f}")

if __name__ == "__main__":
    main()