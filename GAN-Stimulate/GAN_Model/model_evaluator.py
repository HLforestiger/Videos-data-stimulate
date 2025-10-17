import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json
from datetime import datetime

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.figsize'] = (10, 6)  # 默认图表大小
plt.rcParams['figure.dpi'] = 100  # 提高分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图像的分辨率

class ModelEvaluator:
    def __init__(self, real_data_path, generated_data_path, config_path=None, fields=None):
        """
        初始化模型评估器
        
        Args:
            real_data_path: 真实数据路径
            generated_data_path: 生成数据路径
            config_path: 配置文件路径
            fields: 要评估的字段列表
        """
        self.real_data = pd.read_csv(real_data_path)
        self.generated_data = pd.read_csv(generated_data_path)
        self.config = self._load_config(config_path)
        self.fields = fields or self.config.get('data', {}).get('fields', ['likes', 'favorites', 'reposts', 'duration'])
        self.eval_dir = 'evaluation_results'
        os.makedirs(self.eval_dir, exist_ok=True)
        
        # 过滤掉不需要的字段
        self.real_data = self.real_data[[f for f in self.fields if f in self.real_data.columns]]
        self.generated_data = self.generated_data[[f for f in self.fields if f in self.generated_data.columns]]
        
        # 确保两个数据集的字段一致
        common_fields = list(set(self.real_data.columns) & set(self.generated_data.columns))
        self.real_data = self.real_data[common_fields]
        self.generated_data = self.generated_data[common_fields]
        self.fields = common_fields
        
        print(f"已加载数据，真实数据形状: {self.real_data.shape}, 生成数据形状: {self.generated_data.shape}")
        print(f"评估字段: {self.fields}")
    
    def _load_config(self, config_path):
        """加载配置文件"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def calculate_statistical_metrics(self):
        """计算统计指标对比"""
        print("\n=== 统计指标对比 ===")
        metrics = []
        
        for field in self.fields:
            real_values = self.real_data[field]
            generated_values = self.generated_data[field]
            
            # 计算各种统计指标
            stats_real = {
                '字段': field,
                '数据类型': '真实数据',
                '均值': real_values.mean(),
                '标准差': real_values.std(),
                '最小值': real_values.min(),
                '中位数': real_values.median(),
                '最大值': real_values.max(),
                'Q1(25%)': real_values.quantile(0.25),
                'Q3(75%)': real_values.quantile(0.75),
                '偏度': real_values.skew(),
                '峰度': real_values.kurtosis()
            }
            
            stats_gen = {
                '字段': field,
                '数据类型': '生成数据',
                '均值': generated_values.mean(),
                '标准差': generated_values.std(),
                '最小值': generated_values.min(),
                '中位数': generated_values.median(),
                '最大值': generated_values.max(),
                'Q1(25%)': generated_values.quantile(0.25),
                'Q3(75%)': generated_values.quantile(0.75),
                '偏度': generated_values.skew(),
                '峰度': generated_values.kurtosis()
            }
            
            metrics.append(stats_real)
            metrics.append(stats_gen)
        
        stats_df = pd.DataFrame(metrics)
        
        # 计算相对误差
        error_metrics = []
        for field in self.fields:
            real_mean = self.real_data[field].mean()
            gen_mean = self.generated_data[field].mean()
            real_std = self.real_data[field].std()
            gen_std = self.generated_data[field].std()
            
            # 避免除以0
            mean_error = abs(real_mean - gen_mean) / max(1, real_mean) * 100 if real_mean != 0 else float('inf')
            std_error = abs(real_std - gen_std) / max(1, real_std) * 100 if real_std != 0 else float('inf')
            
            error_metrics.append({
                '字段': field,
                '均值相对误差(%)': mean_error,
                '标准差相对误差(%)': std_error
            })
        
        error_df = pd.DataFrame(error_metrics)
        
        # 保存结果
        stats_df.to_csv(f'{self.eval_dir}/statistical_metrics.csv', index=False, encoding='utf-8-sig')
        error_df.to_csv(f'{self.eval_dir}/error_metrics.csv', index=False, encoding='utf-8-sig')
        
        print("统计指标对比结果:")
        print(error_df.round(2))
        
        return stats_df, error_df
    
    def calculate_distribution_similarity(self):
        """计算分布相似度指标"""
        print("\n=== 分布相似度指标 ===")
        similarity_metrics = []
        
        for field in self.fields:
            # 对数值进行对数变换以改善分布
            real_log = np.log1p(self.real_data[field])
            gen_log = np.log1p(self.generated_data[field])
            
            # 计算KL散度
            hist_real, bins = np.histogram(real_log, bins=50, density=True)
            hist_gen, _ = np.histogram(gen_log, bins=bins, density=True)
            
            # 添加小量以避免log(0)
            hist_real = np.maximum(hist_real, 1e-10)
            hist_gen = np.maximum(hist_gen, 1e-10)
            
            kl_div = stats.entropy(hist_real, hist_gen)
            js_dist = jensenshannon(hist_real, hist_gen)
            
            # 计算KS检验
            ks_stat, ks_pvalue = stats.kstest(real_log, gen_log)
            
            similarity_metrics.append({
                '字段': field,
                'KL散度': kl_div,
                'JS距离': js_dist,
                'JS相似度': 1 - js_dist,  # 相似度 = 1 - 距离
                'KS统计量': ks_stat,
                'KS p值': ks_pvalue
            })
        
        similarity_df = pd.DataFrame(similarity_metrics)
        
        # 保存结果
        similarity_df.to_csv(f'{self.eval_dir}/distribution_similarity.csv', index=False, encoding='utf-8-sig')
        
        print("分布相似度指标:")
        print(similarity_df.round(4))
        
        # 解释KS检验结果
        print("\nKS检验解释:")
        for _, row in similarity_df.iterrows():
            if row['KS p值'] > 0.05:
                print(f"{row['字段']}: 无法拒绝原假设，分布相似 (p={row['KS p值']:.4f})")
            else:
                print(f"{row['字段']}: 拒绝原假设，分布有显著差异 (p={row['KS p值']:.4f})")
        
        return similarity_df
    
    def train_discriminator(self):
        """训练判别器评估生成数据的质量"""
        print("\n=== 判别器评估 ===")
        
        # 创建标记数据
        real_labeled = self.real_data.copy()
        real_labeled['label'] = 0  # 真实数据标记为0
        
        gen_labeled = self.generated_data.copy()
        gen_labeled['label'] = 1  # 生成数据标记为1
        
        # 合并数据集
        combined = pd.concat([real_labeled, gen_labeled], ignore_index=True)
        X = combined.drop('label', axis=1)
        y = combined['label']
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        # 训练逻辑回归判别器
        discriminator = LogisticRegression(max_iter=1000, random_state=42)
        discriminator.fit(X_train, y_train)
        
        # 预测
        y_pred = discriminator.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 计算判别分数
        # 如果模型完美，准确率应为100%，但我们希望模型无法区分，所以准确率接近50%更好
        discrimination_score = abs(accuracy - 0.5)  # 越接近0越好
        quality_score = 1 - discrimination_score  # 越接近1越好
        
        print(f"判别器准确率: {accuracy:.4f}")
        print(f"生成数据质量评分: {quality_score:.4f} (越高越好)")
        
        # 保存结果
        with open(f'{self.eval_dir}/discriminator_metrics.txt', 'w', encoding='utf-8') as f:
            f.write(f"判别器准确率: {accuracy:.4f}\n")
            f.write(f"生成数据质量评分: {quality_score:.4f} (越高越好)\n")
            f.write("\n分类报告:\n")
            f.write(classification_report(y_test, y_pred))
        
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        return accuracy, quality_score
    
    def evaluate_correlations(self):
        """评估特征相关性"""
        print("\n=== 特征相关性评估 ===")
        
        # 计算相关系数矩阵
        corr_real = self.real_data.corr()
        corr_gen = self.generated_data.corr()
        
        # 计算相关系数差异
        corr_diff = corr_real - corr_gen
        
        # 计算相关系数误差
        corr_error = {
            '字段对': [],
            '真实相关性': [],
            '生成相关性': [],
            '绝对误差': []
        }
        
        for i in range(len(self.fields)):
            for j in range(i+1, len(self.fields)):
                field1 = self.fields[i]
                field2 = self.fields[j]
                real_corr = corr_real.loc[field1, field2]
                gen_corr = corr_gen.loc[field1, field2]
                error = abs(real_corr - gen_corr)
                
                corr_error['字段对'].append(f"{field1}-{field2}")
                corr_error['真实相关性'].append(real_corr)
                corr_error['生成相关性'].append(gen_corr)
                corr_error['绝对误差'].append(error)
        
        corr_error_df = pd.DataFrame(corr_error)
        avg_error = corr_error_df['绝对误差'].mean()
        
        # 保存相关性矩阵
        corr_real.to_csv(f'{self.eval_dir}/real_correlation.csv', encoding='utf-8-sig')
        corr_gen.to_csv(f'{self.eval_dir}/generated_correlation.csv', encoding='utf-8-sig')
        corr_error_df.to_csv(f'{self.eval_dir}/correlation_error.csv', index=False, encoding='utf-8-sig')
        
        print(f"平均相关性误差: {avg_error:.4f}")
        print("\n特征相关性误差:")
        print(corr_error_df.round(4))
        
        # 绘制相关性热图
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        sns.heatmap(corr_real, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('真实数据相关性矩阵')
        
        plt.subplot(1, 3, 2)
        sns.heatmap(corr_gen, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('生成数据相关性矩阵')
        
        plt.subplot(1, 3, 3)
        sns.heatmap(corr_diff, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('相关性差异矩阵')
        
        plt.tight_layout()
        plt.savefig(f'{self.eval_dir}/correlation_heatmaps.png', bbox_inches='tight')
        plt.close()
        
        return corr_real, corr_gen, corr_diff
    
    def evaluate_nearest_neighbors(self):
        """评估最近邻距离"""
        print("\n=== 最近邻评估 ===")
        
        # 标准化数据
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(self.real_data)
        gen_scaled = scaler.transform(self.generated_data)
        
        # 训练KNN模型
        knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
        knn.fit(real_scaled)
        
        # 计算生成数据到真实数据的最近邻距离
        distances, _ = knn.kneighbors(gen_scaled)
        mean_dist = distances.mean()
        median_dist = np.median(distances)
        
        print(f"生成数据到真实数据的平均最近邻距离: {mean_dist:.4f}")
        print(f"生成数据到真实数据的中位数最近邻距离: {median_dist:.4f}")
        
        # 保存结果
        with open(f'{self.eval_dir}/nearest_neighbor_metrics.txt', 'w', encoding='utf-8') as f:
            f.write(f"生成数据到真实数据的平均最近邻距离: {mean_dist:.4f}\n")
            f.write(f"生成数据到真实数据的中位数最近邻距离: {median_dist:.4f}\n")
        
        # 绘制距离分布
        plt.figure(figsize=(10, 6))
        sns.histplot(distances.mean(axis=1), bins=50, kde=True)
        plt.title('生成数据到真实数据的平均最近邻距离分布')
        plt.xlabel('平均最近邻距离')
        plt.ylabel('频次')
        plt.savefig(f'{self.eval_dir}/nearest_neighbor_distribution.png', bbox_inches='tight')
        plt.close()
        
        return mean_dist, median_dist
    
    def plot_distribution_comparison(self):
        """绘制分布对比图"""
        print("\n=== 绘制分布对比图 ===")
        
        for field in self.fields:
            plt.figure(figsize=(12, 6))
            
            # 对数空间的直方图
            real_log = np.log1p(self.real_data[field])
            gen_log = np.log1p(self.generated_data[field])
            
            # 计算统一的bins
            x_min = min(real_log.min(), gen_log.min()) - 0.5
            x_max = max(real_log.max(), gen_log.max()) + 0.5
            bins = np.linspace(x_min, x_max, 50)
            
            plt.hist(real_log, bins=bins, alpha=0.6, label='真实数据', color='blue')
            plt.hist(gen_log, bins=bins, alpha=0.6, label='生成数据', color='orange')
            plt.title(f'{field} 对数分布对比')
            plt.xlabel(f'log({field} + 1)')
            plt.ylabel('频次')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(f'{self.eval_dir}/{field}_distribution_comparison.png', bbox_inches='tight')
            plt.close()
        
        print(f"分布对比图已保存到 {self.eval_dir} 目录")
    
    def generate_overall_score(self):
        """生成总体评分"""
        print("\n=== 总体评分 ===")
        
        # 读取已保存的指标
        try:
            error_df = pd.read_csv(f'{self.eval_dir}/error_metrics.csv')
            similarity_df = pd.read_csv(f'{self.eval_dir}/distribution_similarity.csv')
            
            # 读取判别器评分
            with open(f'{self.eval_dir}/discriminator_metrics.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                quality_score = float(lines[1].split(':')[1].strip().split()[0])
            
            # 读取相关性误差
            corr_error_df = pd.read_csv(f'{self.eval_dir}/correlation_error.csv')
            avg_corr_error = corr_error_df['绝对误差'].mean()
            
            # 计算统计误差得分 (越低越好，转换为0-1评分)
            # 对每个字段的均值和标准差相对误差取平均
            avg_stat_error = (error_df['均值相对误差(%)'].mean() + error_df['标准差相对误差(%)'].mean()) / 2
            stat_score = max(0, 1 - avg_stat_error / 100)  # 假设100%误差为最差情况
            
            # 分布相似度得分
            avg_js_similarity = similarity_df['JS相似度'].mean()
            
            # 相关性得分
            corr_score = max(0, 1 - avg_corr_error)  # 相关性误差越接近0越好
            
            # 权重可以根据需要调整
            weights = {
                'statistical': 0.25,  # 统计指标权重
                'distribution': 0.25,  # 分布相似度权重
                'discriminator': 0.3,  # 判别器评分权重
                'correlation': 0.2     # 相关性权重
            }
            
            # 计算加权总分
            overall_score = (
                stat_score * weights['statistical'] +
                avg_js_similarity * weights['distribution'] +
                quality_score * weights['discriminator'] +
                corr_score * weights['correlation']
            )
            
            # 保存评分结果
            with open(f'{self.eval_dir}/overall_score.txt', 'w', encoding='utf-8') as f:
                f.write(f"=== 模型整体评分 ===\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"统计指标得分: {stat_score:.4f} (权重: {weights['statistical']})\n")
                f.write(f"分布相似度得分: {avg_js_similarity:.4f} (权重: {weights['distribution']})\n")
                f.write(f"判别器评分: {quality_score:.4f} (权重: {weights['discriminator']})\n")
                f.write(f"相关性得分: {corr_score:.4f} (权重: {weights['correlation']})\n\n")
                f.write(f"最终总体评分: {overall_score:.4f}\n\n")
                
                # 添加评分解释
                f.write("评分解释:\n")
                if overall_score >= 0.8:
                    f.write("优秀: 生成数据质量很高，分布和统计特性与真实数据非常接近")
                elif overall_score >= 0.6:
                    f.write("良好: 生成数据质量较好，但仍有改进空间")
                elif overall_score >= 0.4:
                    f.write("一般: 生成数据质量一般，需要进一步优化模型")
                else:
                    f.write("较差: 生成数据质量较差，建议大幅调整模型参数或架构")
            
            print(f"统计指标得分: {stat_score:.4f}")
            print(f"分布相似度得分: {avg_js_similarity:.4f}")
            print(f"判别器评分: {quality_score:.4f}")
            print(f"相关性得分: {corr_score:.4f}")
            print(f"最终总体评分: {overall_score:.4f}")
            
            # 显示评分解释
            if overall_score >= 0.8:
                print("评分解释: 优秀 - 生成数据质量很高，分布和统计特性与真实数据非常接近")
            elif overall_score >= 0.6:
                print("评分解释: 良好 - 生成数据质量较好，但仍有改进空间")
            elif overall_score >= 0.4:
                print("评分解释: 一般 - 生成数据质量一般，需要进一步优化模型")
            else:
                print("评分解释: 较差 - 生成数据质量较差，建议大幅调整模型参数或架构")
            
            return overall_score
        
        except FileNotFoundError as e:
            print(f"无法计算总体评分: {e}")
            return None
    
    def run_full_evaluation(self):
        """运行完整的评估流程"""
        print(f"开始模型评估，时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 计算统计指标
        self.calculate_statistical_metrics()
        
        # 2. 计算分布相似度
        self.calculate_distribution_similarity()
        
        # 3. 训练判别器
        self.train_discriminator()
        
        # 4. 评估相关性
        self.evaluate_correlations()
        
        # 5. 评估最近邻
        self.evaluate_nearest_neighbors()
        
        # 6. 绘制分布对比图
        self.plot_distribution_comparison()
        
        # 7. 生成总体评分
        overall_score = self.generate_overall_score()
        
        print(f"\n评估完成! 结果已保存到 {self.eval_dir} 目录")
        print(f"总体评分: {overall_score:.4f} (满分1.0)")
        
        return overall_score

def main():
    # 默认路径
    real_data_path = '../Data_Source/douyin_data_preprocessed.csv'
    generated_data_path = '../Data_Source/douyin_data_generated.csv'
    config_path = 'config.json'
    
    # 初始化评估器
    evaluator = ModelEvaluator(real_data_path, generated_data_path, config_path)
    
    # 运行完整评估
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()