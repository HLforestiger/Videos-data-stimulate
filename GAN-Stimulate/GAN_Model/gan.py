import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset
import random

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.figsize'] = (10, 6)  # 默认图表大小
plt.rcParams['figure.dpi'] = 100  # 提高分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图像的分辨率

# 加载配置文件
def load_config(config_path=None):
    """加载JSON配置文件，自动检测配置文件位置"""
    # 如果没有提供路径，使用与脚本同目录下的config.json
    if config_path is None:
        # 获取脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    
    print(f"正在加载配置文件: {config_path}")
    print(f"配置文件是否存在: {os.path.exists(config_path)}")
    if os.path.exists(config_path):
        print(f"配置文件大小: {os.path.getsize(config_path)} 字节")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"成功加载配置文件，配置内容: {config}")
    print(f"模型配置中的epochs值: {config['model']['epochs']}")
    
    return config

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据加载和预处理类
class DataProcessor:
    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.config = config or {}
        self.raw_data = None
        self.processed_data = None
        self.scalers = {}
        # 从配置中获取字段信息，如果没有则使用默认值
        self.fields = self.config.get('fields', ['likes', 'favorites', 'reposts', 'duration'])
        self.date_field = self.config.get('date_field', 'publish_time')
        self.load_data()
    
    def load_data(self):
        """加载原始数据"""
        print(f"正在加载数据: {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        print(f"数据加载完成，共{len(self.raw_data)}行")
        print(f"数据字段: {list(self.raw_data.columns)}")
    
    def preprocess(self):
        """预处理数据，包括归一化"""
        self.processed_data = self.raw_data.copy()
        
        # 对每个数值字段进行预处理
        for field in self.fields:
            # 对数变换处理大数值
            log_data = np.log1p(self.processed_data[field].values.reshape(-1, 1))
            # 使用StandardScaler进行标准化
            self.scalers[field] = StandardScaler()
            scaled_data = self.scalers[field].fit_transform(log_data)
            self.processed_data[f'{field}_scaled'] = scaled_data.flatten()
        
        # 处理日期字段
        self.processed_data['date_ordinal'] = pd.to_datetime(self.processed_data[self.date_field]).map(datetime.toordinal)
        date_scaler = MinMaxScaler()
        self.scalers['date'] = date_scaler
        self.processed_data['date_scaled'] = date_scaler.fit_transform(self.processed_data[['date_ordinal']]).flatten()
        
        print("数据预处理完成")
        return self.processed_data
    
    def inverse_transform(self, field, scaled_values):
        """逆变换生成的值回到原始范围"""
        if field in self.fields:
            # 逆变换到对数空间
            log_values = self.scalers[field].inverse_transform(scaled_values.reshape(-1, 1))
            # 逆对数变换
            return np.expm1(log_values).astype(int).flatten()
        elif field == 'date':
            # 逆变换日期
            date_ordinals = self.scalers['date'].inverse_transform(scaled_values.reshape(-1, 1))
            return [datetime.fromordinal(int(ordinal)).strftime('%Y.%m.%d') for ordinal in date_ordinals.flatten()]
        return scaled_values

# 自定义数据集类
class NumericDataset(Dataset):
    def __init__(self, data, field):
        self.data = data[field].values.reshape(-1, 1).astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

# 生成器网络
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, hidden_dims=None):
        super(Generator, self).__init__()
        # 如果没有提供隐藏层维度，则使用默认值
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        
        # 添加隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
        
        # 输出层 - 使用LeakyReLU而不是Tanh，允许更广泛的输出范围
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.LeakyReLU(0.2))  # 移除Tanh限制，允许更广泛的输出范围
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)

# 判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_dim=1, hidden_dims=None):
        super(Discriminator, self).__init__()
        # 如果没有提供隐藏层维度，则使用默认值
        if hidden_dims is None:
            hidden_dims = [64, 128]
        
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.3))
        
        # 添加隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
        
        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())  # 使用Sigmoid激活函数输出概率
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# GAN训练器类
class GANTrainer:
    def __init__(self, field_name, data_processor, config=None):
        print(f"初始化GANTrainer，传入的config: {config}")
        self.field_name = field_name
        self.data_processor = data_processor
        self.config = config or {}
        print(f"合并后的config: {self.config}")
        
        # 从配置中获取模型参数
        self.latent_dim = self.config.get('latent_dim', 100)
        self.batch_size = self.config.get('batch_size', 32)
        self.epochs = self.config.get('epochs', 500)
        print(f"为字段 {self.field_name} 配置的训练轮数: {self.epochs}")
        self.generator_hidden_dims = self.config.get('generator_hidden_dims', [128, 64])
        self.discriminator_hidden_dims = self.config.get('discriminator_hidden_dims', [64, 128])
        
        # 从配置中获取优化器参数
        self.lr = self.config.get('lr', 0.0002)
        self.betas = tuple(self.config.get('betas', [0.5, 0.999]))
        
        # 创建数据集和数据加载器
        dataset = NumericDataset(data_processor.processed_data, f'{field_name}_scaled')
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 初始化生成器和判别器
        self.generator = Generator(input_dim=self.latent_dim, hidden_dims=self.generator_hidden_dims).to(device)
        self.discriminator = Discriminator(hidden_dims=self.discriminator_hidden_dims).to(device)
        
        # 定义损失函数
        self.adversarial_loss = nn.BCELoss()
        
        # 定义优化器
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)
        
        # 记录损失
        self.g_losses = []
        self.d_losses = []
    
    def train(self):
        """训练GAN模型"""
        print(f"开始训练{self.field_name}的GAN模型...")
        
        for epoch in range(self.epochs):
            for i, real_data in enumerate(self.dataloader):
                # 准备真实和虚假标签
                valid = torch.ones((real_data.size(0), 1), device=device)
                fake = torch.zeros((real_data.size(0), 1), device=device)
                
                # 训练判别器
                self.optimizer_D.zero_grad()
                
                # 真实数据损失
                real_loss = self.adversarial_loss(self.discriminator(real_data.to(device)), valid)
                
                # 生成虚假数据
                z = torch.randn((real_data.size(0), self.latent_dim), device=device)
                fake_data = self.generator(z)
                
                # 虚假数据损失
                fake_loss = self.adversarial_loss(self.discriminator(fake_data.detach()), fake)
                
                # 总判别器损失
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()
                
                # 训练生成器
                self.optimizer_G.zero_grad()
                
                # 生成器损失
                g_loss = self.adversarial_loss(self.discriminator(fake_data), valid)
                g_loss.backward()
                self.optimizer_G.step()
            
            # 记录损失
            self.g_losses.append(g_loss.item())
            self.d_losses.append(d_loss.item())
            
            # 每100个epoch打印一次进度
            if (epoch + 1) % 100 == 0:
                print(f'字段: {self.field_name}, 第 {epoch+1}/{self.epochs} 轮, G损失: {g_loss.item():.4f}, D损失: {d_loss.item():.4f}')
        
        print(f"{self.field_name}的GAN模型训练完成")
        self.plot_losses()
    
    def plot_losses(self):
        """绘制美化的训练损失曲线"""
        # 从配置中获取可视化参数
        plot_dir = self.config.get('plot_dir', 'plots')
        figsize = tuple(self.config.get('figsize', [12, 6]))
        dpi = self.config.get('dpi', 300)
        
        plt.figure(figsize=figsize)
        
        # 使用更美观的颜色和线条样式
        plt.plot(self.g_losses, label='生成器损失', color='#1f77b4', linewidth=2.5, alpha=0.8)
        plt.plot(self.d_losses, label='判别器损失', color='#ff7f0e', linewidth=2.5, alpha=0.8)
        
        # 设置标题和标签，使用更大的字体
        plt.title(f'{self.field_name} - GAN训练损失', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('训练轮次', fontsize=14, labelpad=10)
        plt.ylabel('损失值', fontsize=14, labelpad=10)
        
        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 美化图例
        plt.legend(loc='best', fontsize=12, frameon=True, framealpha=0.9, shadow=True)
        
        # 美化网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加背景色
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        
        # 设置边框样式
        for spine in ax.spines.values():
            spine.set_color('#c0c0c0')
            spine.set_linewidth(1.0)
        
        # 确保保存目录存在
        os.makedirs(plot_dir, exist_ok=True)
        plt.tight_layout()  # 调整布局
        plt.savefig(f'{plot_dir}/{self.field_name}_loss.png', bbox_inches='tight', dpi=dpi)
        plt.close()
    
    def generate(self, num_samples):
        """生成新的数据样本"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((num_samples, self.latent_dim), device=device)
            generated_data = self.generator(z).cpu().numpy()
        
        # 使用数据处理器进行逆变换
        return self.data_processor.inverse_transform(self.field_name, generated_data)

# 日期生成器类（使用统计方法）
class DateGenerator:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.min_date = pd.to_datetime(data_processor.raw_data['publish_time']).min()
        self.max_date = pd.to_datetime(data_processor.raw_data['publish_time']).max()
        self.date_range = (self.max_date - self.min_date).days
    
    def generate(self, num_samples):
        """生成随机日期"""
        generated_dates = []
        for _ in range(num_samples):
            # 随机生成日期偏移
            random_days = random.randint(0, self.date_range)
            # 生成新日期
            new_date = self.min_date + timedelta(days=random_days)
            generated_dates.append(new_date.strftime('%Y.%m.%d'))
        return generated_dates

# 主函数
def main():
    # 加载配置文件
    config = load_config()
    
    # 设置随机种子
    set_seed(config['model']['random_seed'])
    
    # 从配置中获取数据参数
    data_config = config['data']
    model_config = config['model']
    optimizer_config = config['optimizer']
    generation_config = config['generation']
    viz_config = config['visualization']
    
    # 打印加载的配置信息
    print(f"从配置文件加载的epochs: {model_config['epochs']}")
    
    # 合并模型和优化器配置用于传递给训练器
    trainer_config = {
        **model_config,
        **optimizer_config,
        **viz_config
    }
    print(f"传递给训练器的配置: {trainer_config}")
    
    # 初始化数据处理器
    data_processor = DataProcessor(data_config['input_path'], {
        'fields': data_config['fields'],
        'date_field': data_config['date_field']
    })
    data_processor.preprocess()
    
    # 字段列表
    fields = data_config['fields']
    
    # 训练每个字段的GAN模型
    trainers = {}
    for field in fields:
        trainer = GANTrainer(field, data_processor, trainer_config)
        trainer.train()
        trainers[field] = trainer
    
    # 初始化日期生成器
    date_generator = DateGenerator(data_processor)
    
    # 生成新数据
    num_samples = generation_config['num_samples']
    generated_data = {
        'index': range(1, num_samples + 1),
        'publish_time': date_generator.generate(num_samples)
    }
    
    # 为每个字段生成数据
    for field in fields:
        generated_data[field] = trainers[field].generate(num_samples)
    
    # 创建DataFrame
    df_generated = pd.DataFrame(generated_data)
    
    # 保存生成的数据
    os.makedirs(os.path.dirname(data_config['output_path']), exist_ok=True)
    df_generated.to_csv(data_config['output_path'], index=False, encoding='utf-8-sig')
    print(f"生成的数据已保存到: {data_config['output_path']}")
    
    # 可视化生成的数据与真实数据的分布
    for field in fields:
        # 从配置中获取可视化参数
        figsize = tuple(viz_config['figsize'])
        dpi = viz_config['dpi']
        plot_dir = viz_config['plot_dir']
        
        plt.figure(figsize=figsize)
        
        # 计算公共的坐标轴范围
        log_real = np.log1p(data_processor.raw_data[field])
        log_generated = np.log1p(df_generated[field])
        x_min = min(log_real.min(), log_generated.min()) - 0.5
        x_max = max(log_real.max(), log_generated.max()) + 0.5
        
        # 从配置中获取颜色
        real_color = viz_config['real_color']
        generated_color = viz_config['generated_color']
        
        # 绘制真实数据分布
        plt.subplot(1, 2, 1)
        # 先计算统一的bins
        bins = np.linspace(x_min, x_max, 51)  # 50个bin需要51个点
        n_real, bins_real, patches_real = plt.hist(
            log_real, bins=bins, alpha=0.8, color=real_color, 
            edgecolor='white', linewidth=1.2, label='真实数据'
        )
        
        # 设置标题和标签
        plt.title(f'{field} - 真实数据分布(对数空间)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(f'log({field} + 1)', fontsize=14, labelpad=10)
        plt.ylabel('频次', fontsize=14, labelpad=10)
        
        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 设置坐标轴范围
        plt.xlim(x_min, x_max)
        
        # 美化图例
        plt.legend(fontsize=12, frameon=True, framealpha=0.9, shadow=True)
        
        # 美化网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置背景色
        ax1 = plt.gca()
        ax1.set_facecolor('#f8f9fa')
        
        # 设置边框样式
        for spine in ax1.spines.values():
            spine.set_color('#c0c0c0')
            spine.set_linewidth(1.0)
        
        # 绘制生成数据分布
        plt.subplot(1, 2, 2)
        # 使用与真实数据相同的bins
        n_gen, bins_gen, patches_gen = plt.hist(
            log_generated, bins=bins, alpha=0.8, color=generated_color,
            edgecolor='white', linewidth=1.2, label='生成数据'
        )
        
        # 设置标题和标签
        plt.title(f'{field} - 生成数据分布(对数空间)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel(f'log({field} + 1)', fontsize=14, labelpad=10)
        plt.ylabel('频次', fontsize=14, labelpad=10)
        
        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 设置坐标轴范围
        plt.xlim(x_min, x_max)
        
        # 美化图例
        plt.legend(fontsize=12, frameon=True, framealpha=0.9, shadow=True)
        
        # 美化网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置背景色
        ax2 = plt.gca()
        ax2.set_facecolor('#f8f9fa')
        
        # 设置边框样式
        for spine in ax2.spines.values():
            spine.set_color('#c0c0c0')
            spine.set_linewidth(1.0)
        
        # 确保保存目录存在
        os.makedirs(plot_dir, exist_ok=True)
        plt.tight_layout(pad=3.0)
        plt.savefig(f'{plot_dir}/{field}_distribution.png', bbox_inches='tight', dpi=dpi)
        plt.close()
    
    # 添加一个综合对比图
    comparison_figsize = tuple(viz_config['comparison_figsize'])
    plt.figure(figsize=comparison_figsize)
    
    # 为每个字段绘制对比直方图
    for i, field in enumerate(fields):
        plt.subplot(2, 2, i+1)
        
        log_real = np.log1p(data_processor.raw_data[field])
        log_generated = np.log1p(df_generated[field])
        
        # 从配置中获取颜色
        real_color = viz_config['real_color']
        generated_color = viz_config['generated_color']
        
        # 计算统一的bins
        x_min_comp = min(log_real.min(), log_generated.min()) - 0.5
        x_max_comp = max(log_real.max(), log_generated.max()) + 0.5
        bins_comp = np.linspace(x_min_comp, x_max_comp, 41)  # 40个bin需要41个点
        
        # 绘制直方图，使用更细的透明度
        plt.hist(log_real, bins=bins_comp, alpha=0.6, color=real_color, 
                edgecolor='white', linewidth=1.0, label='真实数据')
        plt.hist(log_generated, bins=bins_comp, alpha=0.6, color=generated_color,
                edgecolor='white', linewidth=1.0, label='生成数据')
        
        # 设置标题和标签
        plt.title(f'{field} - 数据分布对比', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(f'log({field} + 1)', fontsize=12, labelpad=8)
        plt.ylabel('频次', fontsize=12, labelpad=8)
        
        # 设置刻度字体大小
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # 美化图例
        plt.legend(fontsize=11, frameon=True, framealpha=0.9)
        
        # 美化网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置背景色
        ax = plt.gca()
        ax.set_facecolor('#f8f9fa')
        
        # 设置边框样式
        for spine in ax.spines.values():
            spine.set_color('#c0c0c0')
            spine.set_linewidth(0.8)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{plot_dir}/all_fields_comparison.png', bbox_inches='tight', dpi=dpi)
    plt.close()
    
    # 显示生成数据的统计信息
    print("\n生成数据统计信息:")
    for field in fields:
        print(f"{field}: 最小值={df_generated[field].min()}, 最大值={df_generated[field].max()}, 平均值={df_generated[field].mean():.2f}")
    
    print("\n生成数据预览:")
    print(df_generated.head())

if __name__ == "__main__":
    main()