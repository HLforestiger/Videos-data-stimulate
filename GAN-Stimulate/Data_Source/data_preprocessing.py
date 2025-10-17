import pandas as pd

def remove_zero_values(input_file, output_file):
    """
    预处理数据，删除包含0值的行
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    
    Returns:
        处理后的DataFrame
    """
    print(f"正在读取数据文件: {input_file}")
    
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 显示原始数据信息
    print("原始数据预览:")
    print(df.head())
    print(f"原始数据形状: {df.shape}")
    print(f"原始记录数: {len(df)}")
    
    # 检查哪些列中包含0值
    zero_counts = {col: (df[col] == 0).sum() for col in df.columns if col != 'index' and col != 'publish_time'}
    print("\n各列0值统计:")
    for col, count in zero_counts.items():
        print(f"{col}: {count} 个0值 ({count/len(df)*100:.2f}%)")
    
    # 找出包含0值的列（排除index和publish_time）
    numeric_columns = [col for col in df.columns if col != 'index' and col != 'publish_time']
    
    # 删除包含0值的行
    initial_rows = len(df)
    df_cleaned = df[(df[numeric_columns] != 0).all(axis=1)]
    removed_rows = initial_rows - len(df_cleaned)
    
    print(f"\n删除了 {removed_rows} 行包含0值的数据")
    print(f"处理后的数据形状: {df_cleaned.shape}")
    print(f"处理后的数据占比: {len(df_cleaned)/initial_rows*100:.2f}%")
    
    # 保存处理后的数据
    df_cleaned.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n处理后的数据已保存到: {output_file}")
    
    # 显示处理后的数据预览
    print("\n处理后的数据预览:")
    print(df_cleaned.head())
    
    return df_cleaned

if __name__ == "__main__":
    # 设置输入和输出文件路径
    input_file = "douyin_data_processed.csv"
    output_file = "douyin_data_preprocessed.csv"
    
    try:
        # 执行数据预处理
        cleaned_df = remove_zero_values(input_file, output_file)
        
        # 显示处理后的统计信息
        print("\n处理后的数据统计信息:")
        numeric_columns = [col for col in cleaned_df.columns if col != 'index' and col != 'publish_time']
        for col in numeric_columns:
            print(f"{col}范围: {cleaned_df[col].min()} - {cleaned_df[col].max()}")
            
    except Exception as e:
        print(f"处理数据时出错: {e}")