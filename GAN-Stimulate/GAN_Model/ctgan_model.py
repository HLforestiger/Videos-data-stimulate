import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
import random
from ctgan import CTGAN

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
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

# 数据加载和预处理类
class DataProcessor:
    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.config = config or {}
        self.raw_data = None
        self.processed_data = None
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
        """预处理数据，为CTGAN准备"""
        # 训练时只使用数值字段，不包含日期字段
        selected_fields = self.fields.copy()
        
        # 复制选定的字段
        self.processed_data = self.raw_data[selected_fields].copy()
        
        # 应用对数变换以改善数据分布，使模型更容易学习
        for field in selected_fields:
            if field in self.processed_data.columns:
                # 加1后取对数，避免log(0)问题
                self.processed_data[f'{field}_log'] = np.log1p(self.processed_data[field])
                # 删除原始字段，只使用对数变换后的值
                self.processed_data.drop(columns=[field], inplace=True)
        
        # 保存字段名映射，用于后续逆变换
        self.log_field_mapping = {f'{field}_log': field for field in selected_fields}
        
        print("数据预处理完成，应用了对数变换以改善数据分布")
        print(f"处理后的数据形状: {self.processed_data.shape}")
        print(f"处理后的数据字段: {list(self.processed_data.columns)}")
        return self.processed_data

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

# CTGAN训练器类
class CTGANTrainer:
    def __init__(self, data_processor, config=None):
        self.data_processor = data_processor
        self.config = config or {}
        
        # 从配置中获取模型参数
        self.epochs = self.config.get('epochs', 300)
        self.batch_size = self.config.get('batch_size', 500)
        self.embedding_dim = self.config.get('embedding_dim', 128)
        self.gen_dim = self.config.get('generator_hidden_dims', (256, 256))
        self.dis_dim = self.config.get('discriminator_hidden_dims', (256, 256))
        self.lr = self.config.get('lr', 0.0002)
        
        # 不需要分类字段，因为我们只使用数值字段进行训练
        self.categorical_columns = []
        
        # 初始化CTGAN模型 (使用最新版本的API)
        # 设置pac=1，避免batch_size必须是pac整数倍的限制
        # 增加epochs和调整batch_size以提高模型质量
        self.model = CTGAN(
            batch_size=min(500, len(self.data_processor.processed_data)),  # 动态调整batch_size
            epochs=self.epochs,
            verbose=True,
            pac=1,
            generator_lr=0.0002,
            discriminator_lr=0.0002
        )
    
    def train(self):
        """训练CTGAN模型"""
        print(f"开始训练CTGAN模型，共{self.epochs}轮...")
        
        # 训练模型
        self.model.fit(
            self.data_processor.processed_data,
            self.categorical_columns
        )
        
        print("CTGAN模型训练完成")
    
    def generate(self, num_samples):
        """生成新的数据样本，并确保数据合理"""
        print(f"生成{num_samples}条新数据...")
        generated_data = self.model.sample(num_samples)
        
        # 创建结果DataFrame
        result_data = pd.DataFrame()
        
        # 检查是否有对数变换的字段需要逆变换
        if hasattr(self.data_processor, 'log_field_mapping'):
            for log_field, original_field in self.data_processor.log_field_mapping.items():
                if log_field in generated_data.columns:
                    # 逆变换：应用expm1并向上取整到至少1
                    transformed_values = np.expm1(generated_data[log_field])
                    # 使用随机值替换1，避免过多的1值
                    # 对于接近0的值，使用1-5之间的随机数
                    min_values = np.random.randint(1, 6, size=len(transformed_values))
                    # 确保所有值都大于0
                    result_data[original_field] = np.maximum(transformed_values, min_values)
                    
                    # 应用合理的上限
                    if original_field in self.data_processor.raw_data.columns:
                        max_val = self.data_processor.raw_data[original_field].max() * 1.2
                        result_data[original_field] = result_data[original_field].clip(upper=max_val)
                    
                    # 对于duration字段，设置额外限制
                    if original_field == 'duration':
                        result_data[original_field] = result_data[original_field].clip(upper=8000)
        else:
            # 如果没有对数变换，则使用原来的字段
            result_data = generated_data.copy()
            # 确保没有0值，使用1-3之间的随机数
            for field in result_data.columns:
                mask = result_data[field] <= 1
                result_data.loc[mask, field] = np.random.randint(1, 4, size=mask.sum())
        
        print("数据后处理完成，应用了逆变换并避免过多的1值")
        return result_data

# 可视化函数
def visualize_distributions(data_processor, generated_data, viz_config):
    """可视化生成数据与真实数据的分布对比"""
    fields = data_processor.fields
    plot_dir = viz_config['plot_dir']
    figsize = tuple(viz_config['figsize'])
    dpi = viz_config['dpi']
    real_color = viz_config['real_color']
    generated_color = viz_config['generated_color']
    
    # 确保保存目录存在
    os.makedirs(plot_dir, exist_ok=True)
    
    # 为每个字段绘制分布对比图
    for field in fields:
        plt.figure(figsize=figsize)
        
        # 计算公共的坐标轴范围
        log_real = np.log1p(data_processor.raw_data[field])
        log_generated = np.log1p(generated_data[field])
        x_min = min(log_real.min(), log_generated.min()) - 0.5
        x_max = max(log_real.max(), log_generated.max()) + 0.5
        
        # 计算统一的bins
        bins = np.linspace(x_min, x_max, 51)  # 50个bin需要51个点
        
        # 绘制真实数据分布
        plt.subplot(1, 2, 1)
        plt.hist(
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
        plt.hist(
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
        log_generated = np.log1p(generated_data[field])
        
        # 计算统一的bins
        x_min_comp = min(log_real.min(), log_generated.min()) - 0.5
        x_max_comp = max(log_real.max(), log_generated.max()) + 0.5
        bins_comp = np.linspace(x_min_comp, x_max_comp, 41)  # 40个bin需要41个点
        
        # 绘制直方图
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
    
    # 合并模型和优化器配置用于传递给训练器
    trainer_config = {
        **model_config,
        **optimizer_config
    }
    
    # 打印配置信息
    print(f"从配置文件加载的epochs: {model_config['epochs']}")
    print(f"传递给训练器的配置: {trainer_config}")
    
    # 初始化数据处理器
    data_processor = DataProcessor(data_config['input_path'], {
        'fields': data_config['fields'],
        'date_field': data_config['date_field']
    })
    data_processor.preprocess()
    
    # 初始化CTGAN训练器
    trainer = CTGANTrainer(data_processor, trainer_config)
    
    # 训练模型
    trainer.train()
    
    # 生成新数据
    num_samples = generation_config['num_samples']
    generated_data = trainer.generate(num_samples)
    
    # 初始化日期生成器（如果需要重新生成日期）
    date_generator = DateGenerator(data_processor)
    
    # 如果需要，可以替换生成的日期
    if data_config['date_field'] in generated_data.columns:
        generated_data[data_config['date_field']] = date_generator.generate(num_samples)
    else:
        # 添加索引和日期字段
        generated_data['index'] = range(1, num_samples + 1)
        generated_data['publish_time'] = date_generator.generate(num_samples)
    
    # 确保所有字段都存在
    for field in data_config['fields']:
        if field not in generated_data.columns:
            print(f"警告: 字段 {field} 不在生成的数据中")
    
    # 保存生成的数据
    os.makedirs(os.path.dirname(data_config['output_path']), exist_ok=True)
    generated_data.to_csv(data_config['output_path'], index=False, encoding='utf-8-sig')
    print(f"生成的数据已保存到: {data_config['output_path']}")
    
    # 可视化生成的数据与真实数据的分布
    visualize_distributions(data_processor, generated_data, viz_config)
    
    # 显示生成数据的统计信息
    print("\n生成数据统计信息:")
    for field in data_config['fields']:
        if field in generated_data.columns:
            print(f"{field}: 最小值={generated_data[field].min()}, 最大值={generated_data[field].max()}, 平均值={generated_data[field].mean():.2f}")
    
    print("\n生成数据预览:")
    print(generated_data.head())

if __name__ == "__main__":
    main()