import pandas as pd
import re
from datetime import datetime, timedelta
import numpy as np

def convert_wan_to_number(value):
    """将包含'万'的数字转换为正常数字"""
    if pd.isna(value) or value == '':
        return 0
    
    # 转换为字符串处理
    value_str = str(value)
    
    # 如果包含'万'，则乘以10000
    if '万' in value_str:
        # 提取数字部分
        number_match = re.search(r'(\d+\.?\d*)', value_str)
        if number_match:
            number = float(number_match.group(1))
            return int(number * 10000)
    
    # 如果不包含'万'，直接转换为数字
    try:
        return int(float(value_str))
    except (ValueError, TypeError):
        return 0

def convert_duration_to_seconds(duration_str):
    """将时长转换为秒数"""
    if pd.isna(duration_str) or duration_str == '':
        return 0
    
    duration_str = str(duration_str).strip()
    
    # 处理不同格式的时长
    if ':' in duration_str:
        parts = duration_str.split(':')
        if len(parts) == 2:  # MM:SS 格式
            minutes, seconds = parts
            return int(minutes) * 60 + int(seconds)
        elif len(parts) == 3:  # HH:MM:SS 格式
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    
    # 如果只是数字，假设是秒
    try:
        return int(float(duration_str))
    except (ValueError, TypeError):
        return 0

def convert_publish_time_to_date(publish_time_str):
    """将发布时间转换为具体日期"""
    if pd.isna(publish_time_str) or publish_time_str == '':
        return None
    
    publish_time_str = str(publish_time_str).strip()
    
    # 移除开头的"· "符号
    if publish_time_str.startswith('· '):
        publish_time_str = publish_time_str[2:]
    
    current_date = datetime.now()
    
    # 处理相对时间
    if '小时前' in publish_time_str:
        hours = int(re.search(r'(\d+)', publish_time_str).group(1))
        return (current_date - timedelta(hours=hours)).strftime('%Y.%m.%d')
    
    elif '天前' in publish_time_str:
        days = int(re.search(r'(\d+)', publish_time_str).group(1))
        return (current_date - timedelta(days=days)).strftime('%Y.%m.%d')
    
    elif '周前' in publish_time_str:
        weeks = int(re.search(r'(\d+)', publish_time_str).group(1))
        return (current_date - timedelta(weeks=weeks)).strftime('%Y.%m.%d')
    
    # 处理具体日期
    elif '月' in publish_time_str and '日' in publish_time_str:
        # 格式如 "9月15日"
        month_match = re.search(r'(\d+)月', publish_time_str)
        day_match = re.search(r'(\d+)日', publish_time_str)
        
        if month_match and day_match:
            month = int(month_match.group(1))
            day = int(day_match.group(1))
            # 假设是当前年份
            year = current_date.year
            return f"{year}.{month:02d}.{day:02d}"
    
    # 如果无法解析，返回原字符串
    return publish_time_str

def process_data(input_file, output_file):
    """处理数据文件"""
    print("正在读取数据文件...")
    df = pd.read_csv(input_file)
    
    print("原始数据预览:")
    print(df.head())
    print(f"数据形状: {df.shape}")
    
    # 处理likes列
    print("正在处理likes列...")
    df['likes'] = df['likes'].apply(convert_wan_to_number)
    
    # 处理favorites列
    print("正在处理favorites列...")
    df['favorites'] = df['favorites'].apply(convert_wan_to_number)
    
    # 处理reposts列
    print("正在处理reposts列...")
    df['reposts'] = df['reposts'].apply(convert_wan_to_number)
    
    # 处理duration列
    print("正在处理duration列...")
    df['duration'] = df['duration'].apply(convert_duration_to_seconds)
    
    # 处理publish_time列
    print("正在处理publish_time列...")
    df['publish_time'] = df['publish_time'].apply(convert_publish_time_to_date)
    
    print("处理完成！")
    print("处理后的数据预览:")
    print(df.head())
    
    # 保存处理后的数据
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"处理后的数据已保存到: {output_file}")
    
    return df

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file = "Data_Source/douyin_data_source.csv"
    output_file = "Data_Source/douyin_data_processed.csv"
    
    try:
        # 处理数据
        processed_df = process_data(input_file, output_file)
        
        # 显示统计信息
        print("\n数据统计信息:")
        print(f"总记录数: {len(processed_df)}")
        print(f"likes范围: {processed_df['likes'].min()} - {processed_df['likes'].max()}")
        print(f"favorites范围: {processed_df['favorites'].min()} - {processed_df['favorites'].max()}")
        print(f"reposts范围: {processed_df['reposts'].min()} - {processed_df['reposts'].max()}")
        print(f"duration范围: {processed_df['duration'].min()} - {processed_df['duration'].max()} 秒")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
