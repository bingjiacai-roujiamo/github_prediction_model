#########################1.计算相关系数#########################
# 计算相关系数，相关系数越大，说明特征之间的相关性越强
# 相关系数的计算方法有很多种，最常用的是皮尔逊相关系数和斯皮尔曼等级相关系数。


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency

# 读取数据
df = pd.read_csv(r'D:/vscode_work/mechine learning/data_im_删e阳无log+比值.csv')

# 打印列名以检查
print(df.columns)

# 定义二分类列和数值列
binary_cols = ['outcome', 'history', 'sex', 'treatment', 'DNA']
numeric_cols = [col for col in df.columns if col not in binary_cols]

# 初始化相关系数矩阵
corr = pd.DataFrame(np.ones((len(df.columns), len(df.columns))), 
                    index=df.columns, columns=df.columns)

# 分块计算相关系数
for i in range(len(df.columns)):
    for j in range(i+1, len(df.columns)):
        col1 = df.columns[i]
        col2 = df.columns[j]
        
        # 数值型 vs 数值型 → 皮尔逊
        if col1 in numeric_cols and col2 in numeric_cols:
            corr.loc[col1, col2] = df[[col1, col2]].corr().iloc[0,1]
        
        # 数值型 vs 二分类 → 点二列
        elif (col1 in numeric_cols and col2 in binary_cols) or (col1 in binary_cols and col2 in numeric_cols):
            r, _ = pointbiserialr(df[col1], df[col2])
            corr.loc[col1, col2] = r
        
        # 二分类 vs 二分类 → Phi系数
        elif col1 in binary_cols and col2 in binary_cols:
            contingency = pd.crosstab(df[col1], df[col2])
            chi2, _, _, _ = chi2_contingency(contingency)
            n = contingency.sum().sum()
            phi = np.sqrt(chi2 / n)
            corr.loc[col1, col2] = phi
        
        corr.loc[col2, col1] = corr.loc[col1, col2]  # 保持对称性

# 创建图形
fig, ax = plt.subplots(figsize=(16, 14), dpi=600) # 创建图形对象fig和一个坐标轴ax
cmap = plt.cm.viridis # viridis作为颜色映射是一种从黄绿色到深蓝色的颜色渐变
norm = plt.Normalize(vmin=-1, vmax=1) # 创建一个归一化对象，将相关系数的值范围从-1到1映射到颜色映射的范围，以便根据相关系数的大小确定颜色

# 初始化一个空的可绘制兑现用于颜色条
scatter_handles = []

# 循环绘制气泡图和数值
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if i > j: # 对角线左下部分，只显示气泡
            color = cmap(norm(corr.iloc[i, j])) # 根据相关系数获取颜色
            scatter = ax.scatter(i, j, s=np.abs(corr.iloc[i, j])*1000, color=color, alpha=0.75) # 根据相关系数大小绘制气泡大小
            scatter_handles.append(scatter) # 保存scatter对象用于颜色条
        elif i < j: # 对角线右上部分只显示数值
            color = cmap(norm(corr.iloc[i, j])) # 数值的颜色同样基于相关系数
            ax.text(i, j, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color=color, fontsize=10)
        else: # 对角线部分显示空白
            ax.scatter(i, j, s=1, color='white')

# 设置坐标轴标签，要一起运行
ax.set_xticks(range(len(corr.columns))) # 设置X轴刻度，与相应的列和行对应
ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10) # X轴标签旋转45度
ax.set_yticks(range(len(corr.columns))) # 设置Y轴刻度，与相应的列和行对应
ax.set_yticklabels(corr.columns, fontsize=10) # Y轴刻度标签
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) # 创建一个ScalarMappable对象，用于生成颜色条，它使用与气泡图相同的颜色映射和归一化对象
sm.set_array([]) # 设置一个空数组，这是创建颜色条时的一个必要步骤，虽然这里传入的是空数组，但不影响颜色条的生成
fig.colorbar(sm, ax=ax, label='Correlation Coefficient') # 在图形fig中为坐标轴ax添加颜色条，并设置颜色条的标签为“Correlation Coefficient”，颜色条的颜色和刻度根据ScalarMappable对象的设置自动生成，反映了相关系数的大小与颜色的对应关系

# 导出图形
plt.savefig("D:/vscode_work/mechine learning/相关性热图.png", format='png', bbox_inches='tight')
plt.tight_layout() # 自动调整图形的布局，使各个元素（如坐标轴标签、颜色条等）之间保持合适的间距，避免出现重叠或被截断的情况。
plt.show()

# 提取相关系数大于0.7的特征对
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        col1 = corr.columns[i]
        col2 = corr.columns[j]
        if abs(corr.iloc[i, j]) > 0.7:
            high_corr_pairs.append((col1, col2, corr.iloc[i, j]))

# 打印相关系数大于0.7的特征对
print("相关系数大于0.7的特征对:")
for pair in high_corr_pairs:
    print(f"{pair[0]} 和 {pair[1]}: 相关系数 = {pair[2]:.2f}")



#########################2.计算VIF值#########################
# 计算VIF值，VIF值越大，说明特征之间的共线性越严重
# 多重共线性
# pip install statsmodels 在右下powershell中安装


import pandas as pd
import statsmodels.api as sm # pip install statsmodels 在右下powershell中安装
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据
df = pd.read_csv('D:/vscode_work/mechine learning/data_im_删e阳无log+比值.csv')

# 选择特征列
features = ['sex', 'treatment', 'DNA'] + [col for col in df.columns if col not in ['sex', 'treatment', 'DNA', 'outcome']]

# 创建特征矩阵
X = df[features]

# 添加常数项
X = sm.add_constant(X)

# 计算VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#按VIF值从大到小排序
vif_data = vif_data.sort_values(by='VIF', ascending=False)

# 显示VIF结果
print(vif_data)