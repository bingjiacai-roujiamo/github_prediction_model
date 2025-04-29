# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, roc_curve, auc, precision_recall_curve,
                            roc_auc_score, average_precision_score, classification_report)
from sklearn.calibration import calibration_curve
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import joblib
from matplotlib.colors import ListedColormap
import matplotlib.gridspec as gridspec
from itertools import cycle
import upsetplot as up
from sklearn.utils import resample
from bayes_opt import BayesianOptimization
import os
import warnings
warnings.filterwarnings('ignore')

###################################1.数据预处理#######################################

# 设置随机种子，确保结果可重复
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 读取数据
df = pd.read_csv(r'D:/vscode_work/mechine learning/data_im_e阴log.csv')
print("数据集形状:", df.shape)
print("数据集预览:\n", df.head())

# 检查数据是否有缺失值
print("\n缺失值数量:\n", df.isnull().sum())

# 确定特征和目标变量
X = df.drop('outcome', axis=1)

y = df['outcome']

# 确定分类变量
categorical_features = ['sex', 'history', 'treatment', 'DNA']
numeric_features = [col for col in X.columns if col not in categorical_features]

# ①数据预处理：划分训练集和测试集，然后进行标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y)
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 标准化数值特征
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])

# 保存scaler对象
joblib.dump(scaler, 'D:/vscode_work/mechine learning/scaler.pkl')

print("标准化后的训练集预览:\n", X_train_scaled.head())

# 保存训练集和测试集
joblib.dump((X_train_scaled, y_train, X_test_scaled, y_test), 'D:/vscode_work/mechine learning/train_test_data.pkl')

###################################2.特征筛选#######################################

# ②特征筛选：使用七种方法
print("\n开始特征筛选...")

# 用于存储不同方法选择的特征
selected_features = {}

# 1. 逐步回归 - 前向选择 (FS)   我试过了，逐步回归就筛选出来一个变量，所以就注释掉了，你也可以试试
# print("进行前向特征选择...")
# sfs_forward = SFS(
#     estimator=LogisticRegression(random_state=RANDOM_STATE),
#     k_features='best',
#     forward=True,
#     floating=False,
#     scoring='roc_auc',
#     cv=5,
#     n_jobs=-1
# )
# sfs_forward.fit(X_train_scaled, y_train)
# selected_features['Forward Selection'] = list(X_train_scaled.columns[list(sfs_forward.k_feature_idx_)])
# print(f"前向选择选中的特征: {selected_features['Forward Selection']}")

# 2. 逐步回归 - 向后选择 (BS)
# print("进行向后特征选择...")
# sfs_backward = SFS(
#     estimator=LogisticRegression(random_state=RANDOM_STATE),
#     k_features='best',
#     forward=False,
#     floating=False,
#     scoring='roc_auc',
#     cv=5,
#     n_jobs=-1
# )
# sfs_backward.fit(X_train_scaled, y_train)
# selected_features['Backward Selection'] = list(X_train_scaled.columns[list(sfs_backward.k_feature_idx_)])
# print(f"向后选择选中的特征: {selected_features['Backward Selection']}")

# 3. 逐步回归 - 双向消除 (BE)
# print("进行双向特征选择...")
# sfs_bidirectional = SFS(
#     estimator=LogisticRegression(random_state=RANDOM_STATE),
#     k_features='best',
#     forward=True,
#     floating=True,
#     scoring='roc_auc',
#     cv=5,
#     n_jobs=-1
# )
# sfs_bidirectional.fit(X_train_scaled, y_train)
# selected_features['Bidirectional Elimination'] = list(X_train_scaled.columns[list(sfs_bidirectional.k_feature_idx_)])
# print(f"双向消除选中的特征: {selected_features['Bidirectional Elimination']}")

# 4. LASSO回归
print("进行LASSO特征选择...")
lasso = Lasso(alpha=0.056, random_state=RANDOM_STATE)
lasso.fit(X_train_scaled, y_train)
selected_features['LASSO'] = list(X_train_scaled.columns[lasso.coef_ != 0])
print(f"LASSO选中的特征: {selected_features['LASSO']}")

# 5. 随机森林 - 平均降低准确度 (MDA)
rf_mda = RandomForestClassifier(n_estimators=5, random_state=RANDOM_STATE)
rf_mda.fit(X_train_scaled, y_train)
importances_mda = rf_mda.feature_importances_
indices_mda = np.argsort(importances_mda)[::-1]
selected_features_mda = []
cumulative_importance = 0
threshold = 0.95 # 选择累积重要性达到95%的特征

for idx in indices_mda: 
    cumulative_importance += importances_mda[idx]
    selected_features_mda.append(X_train_scaled.columns[idx])
    if cumulative_importance >= threshold:
        break

selected_features['RF_MDA'] = selected_features_mda
print(f"RF_MDA选中的特征: {selected_features['RF_MDA']}")

# 6. 随机森林 - 平均降低基尼杂质 (MDG)
print("使用随机森林 MDG 进行特征选择...")
rf_mdg = RandomForestClassifier(n_estimators=5, criterion='gini', random_state=RANDOM_STATE)
rf_mdg.fit(X_train_scaled, y_train)
importances_mdg = rf_mdg.feature_importances_
indices_mdg = np.argsort(importances_mdg)[::-1]
selected_features_mdg = []
cumulative_importance = 0

for idx in indices_mdg:
    cumulative_importance += importances_mdg[idx]
    selected_features_mdg.append(X_train_scaled.columns[idx])
    if cumulative_importance >= threshold:
        break

selected_features['RF_MDG'] = selected_features_mdg
print(f"RF_MDG选中的特征: {selected_features['RF_MDG']}")

# 7. 支持向量机递归特征消除 (SVM-RFE)
print("进行SVM-RFE特征选择...")
svm_rfe = RFECV(
    estimator=SVC(kernel='linear', random_state=RANDOM_STATE),
    step=1,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
svm_rfe.fit(X_train_scaled, y_train)
selected_features['SVM_RFE'] = list(X_train_scaled.columns[svm_rfe.support_])
print(f"SVM-RFE选中的特征: {selected_features['SVM_RFE']}")

# 8. Logistic Regression RFE  一样的只有一个特征
# print("进行Logistic Regression-RFE特征选择...")
# lr_rfe = RFECV(
#     estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
#     step=1,
#     cv=5,
#     scoring='roc_auc',
#     n_jobs=-1
# )
# lr_rfe.fit(X_train_scaled, y_train)
# selected_features['LR_RFE'] = list(X_train_scaled.columns[lr_rfe.support_])
# print(f"LR-RFE选中的特征: {selected_features['LR_RFE']}")

# 9. Random Forest RFE (more systematic than MDG/MDA approaches)
print("进行Random Forest-RFE特征选择...")
rf_rfe = RFECV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    step=1,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
rf_rfe.fit(X_train_scaled, y_train)
selected_features['RF_RFE'] = list(X_train_scaled.columns[rf_rfe.support_])
print(f"RF-RFE选中的特征: {selected_features['RF_RFE']}")

# 提取可视化数据
n_features = rf_rfe.n_features_  # 最佳特征数量
auc_scores = rf_rfe.cv_results_['mean_test_score']  # 平均AUC值
feature_counts = np.arange(1, len(auc_scores)+1)  # 特征数量序列（从少到多）

# 创建可视化画布
plt.figure(figsize=(10, 6), dpi=100)

# 绘制主曲线
line = plt.plot(feature_counts, auc_scores, 
                marker='o', 
                linestyle='--', 
                color='#2c7fb8',
                linewidth=2,
                markersize=8,
                markerfacecolor='#ff7f0e')

# 标注最佳点
plt.scatter(n_features, auc_scores[feature_counts == n_features], 
            s=120, 
            zorder=10,
            color='#d62728',
            edgecolors='black',
            label=f'Optimal Features Number: {n_features}')

# 添加辅助元素
plt.title("RF-RFE Feature Selection", fontsize=14, pad=20)
plt.xlabel("Features Number", fontsize=12)
plt.ylabel("Cross-Validation AUC", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 优化坐标轴显示
max_features = len(auc_scores)
plt.xticks(np.arange(1, max_features+1, step=1))

# 添加数据标签
for i, (count, score) in enumerate(zip(feature_counts, auc_scores)):
    if i % 2 == 0 or count == n_features:  # 隔行显示标签
        plt.text(count, score+0.005, 
                f"{score:.3f}", 
                ha='center', 
                fontsize=8,
                color='#2ca02c')
plt.savefig('D:/vscode_work/mechine learning/rf_rfe_feature_selection.png', dpi=300, bbox_inches='tight')
plt.show()


# 10. Gradient Boosting RFE
print("进行Gradient Boosting-RFE特征选择...")
gb_rfe = RFECV(
    estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
    step=1,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
gb_rfe.fit(X_train_scaled, y_train)
selected_features['GB_RFE'] = list(X_train_scaled.columns[gb_rfe.support_])
print(f"GB-RFE选中的特征: {selected_features['GB_RFE']}")

# 11.Boruta特征选择
from boruta import BorutaPy
from collections import defaultdict

print("进行Boruta特征选择（重复20次）...")

boruta_rankings = defaultdict(list)

# Boruta 运行 20 次
for i in range(20):
    print(f"  Boruta第{i+1}/20次运行...")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=RANDOM_STATE + i)
    boruta_selector = BorutaPy(estimator=rf, n_estimators='auto', verbose=0, random_state=RANDOM_STATE + i)
    boruta_selector.fit(X_train_scaled.values, y_train.values)

    # 获取每个特征的排名
    rankings = boruta_selector.ranking_
    for feature, rank in zip(X_train_scaled.columns, rankings):
        boruta_rankings[feature].append(rank)

# 计算每个特征在20次中的排名中位数
boruta_median_ranks = {feature: np.median(ranks) for feature, ranks in boruta_rankings.items()}

# 按中位数排序，取前5个特征
sorted_boruta = sorted(boruta_median_ranks.items(), key=lambda x: x[1])
boruta_top5_features = [feature for feature, _ in sorted_boruta[:5]]
selected_features['Boruta'] = boruta_top5_features
print(f"Boruta选中的特征: {selected_features['Boruta']}")

# 找出在5种及以上方法中交集出现的特征
print("\n计算特征交集...")
all_features = list(X_train_scaled.columns)
feature_count = {feature: 0 for feature in all_features}

for method, features in selected_features.items():
    for feature in features:
        feature_count[feature] += 1

final_selected_features = [feature for feature, count in feature_count.items() if count >= 5]
print(f"最终选中的特征 (出现在5种及以上方法中): {final_selected_features}")

# 使用UpSet图进行可视化
# feature_presence = pd.DataFrame(0, index=all_features, columns=selected_features.keys())
# for method, features in selected_features.items():
#     feature_presence.loc[features, method] = 1

# # 转换为集合格式
# feature_sets = {}
# for method in selected_features:
#     feature_sets[method] = set(selected_features[method])

# # 创建UpSet图
# plt.figure(figsize=(12, 8))
# up.plot(feature_sets, sort_by='cardinality', show_counts=True)
# plt.title('Interactions between the predictors (UpSet Plot)')
# plt.savefig('D:/vscode_work/mechine learning/upset_plot.png', dpi=300, bbox_inches='tight')
# plt.close()

# 筛选训练集和测试集，只保留最终选择的特征
final_selected_features = selected_features['Boruta']  # 选择LASSO特征，如果后面不用就注释掉下面两行
print(f"最终选中的特征: {final_selected_features}")
# 手动移除不需要的特征 'ANC'
final_selected_features = [feature for feature in final_selected_features if feature != 'ANC']
print(f"移除 'ANC' 后的最终选中的特征: {final_selected_features}")

X_train_selected = X_train_scaled[final_selected_features]
X_test_selected = X_test_scaled[final_selected_features]

# 保存最终选择的特征和数据
joblib.dump(final_selected_features, 'D:/vscode_work/mechine learning/final_selected_features.pkl')
joblib.dump((X_train_selected, y_train, X_test_selected, y_test), 'D:/vscode_work/mechine learning/selected_train_test_data.pkl')

print(f"最终选择的特征数量: {len(final_selected_features)}")
print("特征选择完成。\n")

###################################3.构建模型#######################################

# ③构建模型
print("开始构建和训练模型...")

# 定义评估函数
def evaluate_model(model, X, y, cv=5):
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return cv_scores.mean(), cv_scores.std()

# 用于存储所有模型
models = {}
model_performances = {}

# 进行贝叶斯超参数优化的通用函数  后面没用到，注释掉了
# def bayesian_optimize(model_class, param_bounds, X, y, init_points=5, n_iter=25, cv=5):
#     def objective(**params):
#         # 处理整数参数
#         for param, value in params.items():
#             if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'n_neighbors', 'hidden_layer_sizes_1', 'hidden_layer_sizes_2']:
#                 params[param] = int(value)
        
#         # 特殊处理隐藏层大小
#         if 'hidden_layer_sizes_1' in params and 'hidden_layer_sizes_2' in params:
#             hidden_layer_sizes_1 = int(params.pop('hidden_layer_sizes_1'))
#             hidden_layer_sizes_2 = int(params.pop('hidden_layer_sizes_2'))
#             params['hidden_layer_sizes'] = (hidden_layer_sizes_1, hidden_layer_sizes_2)
        
#         model = model_class(**params)
#         cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
#         return cv_scores.mean()
    
#     optimizer = BayesianOptimization(
#         f=objective,
#         pbounds=param_bounds,
#         random_state=RANDOM_STATE
#     )
    
#     optimizer.maximize(init_points=init_points, n_iter=n_iter)
#     return optimizer.max

# 1. 逻辑回归 (LR)
print("\n1. 训练逻辑回归模型...")
# 使用标准参数的逻辑回归
lr_model = LogisticRegression(
    penalty='l2',          # 标准L2正则化
    C=1.0,                 # 默认正则化强度
    random_state=RANDOM_STATE,
    max_iter=1000,         # 增加迭代次数以确保收敛
    solver='lbfgs'        # 高效求解器，适合中小规模数据
)

# 在训练集上训练模型
lr_model.fit(X_train_selected, y_train)

# 使用交叉验证评估模型性能
lr_cv_score, lr_cv_std = evaluate_model(lr_model, X_train_selected, y_train)
print(f"逻辑回归 - 5折交叉验证平均AUC: {lr_cv_score:.4f} ± {lr_cv_std:.4f}")


# 保存模型
models['LR'] = lr_model
model_performances['LR'] = {
    'auc': lr_cv_score,
    'std': lr_cv_std,
    'params': {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
}

# 2. 决策树 (DT)
print("\n2. 训练决策树模型...")
# 默认参数
dt_default = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_default_score, dt_default_std = evaluate_model(dt_default, X_train_selected, y_train)
print(f"决策树 (默认参数) - 5折交叉验证平均AUC: {dt_default_score:.4f} ± {dt_default_std:.4f}")

# 贝叶斯超参数优化
dt_param_bounds = {
    'max_depth': (1, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'criterion': ['gini', 'entropy']
}

# 分别对不同的criterion进行优化
dt_best_params = {}
dt_best_score = 0

for criterion in ['gini', 'entropy']:
    param_bounds = {
        'max_depth': (1, 10),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 10)
    }
    
    model_params = {
        'criterion': criterion,
        'random_state': RANDOM_STATE
    }
    
    def objective(**params):
        for param in ['max_depth', 'min_samples_split', 'min_samples_leaf']:
            params[param] = int(params[param])
        model = DecisionTreeClassifier(**{**model_params, **params})
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
        return cv_scores.mean()
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_bounds,
        random_state=RANDOM_STATE
    )
    
    optimizer.maximize(init_points=5, n_iter=15)
    
    if optimizer.max['target'] > dt_best_score:
        dt_best_score = optimizer.max['target']
        dt_best_params = {**model_params, **optimizer.max['params']}
        for param in ['max_depth', 'min_samples_split', 'min_samples_leaf']:
            dt_best_params[param] = int(dt_best_params[param])

print(f"最佳决策树参数: {dt_best_params}")
print(f"决策树 (调优参数) - 5折交叉验证平均AUC: {dt_best_score:.4f}")

# 训练最佳模型
dt_best = DecisionTreeClassifier(**dt_best_params)
dt_best.fit(X_train_selected, y_train)
models['DT'] = dt_best
model_performances['DT'] = {'default_auc': dt_default_score, 'optimized_auc': dt_best_score, 'params': dt_best_params}

# 3. 随机森林 (RF)
print("\n3. 训练随机森林模型...")
# 默认参数
rf_default = RandomForestClassifier(random_state=RANDOM_STATE)
rf_default_score, rf_default_std = evaluate_model(rf_default, X_train_selected, y_train)
print(f"随机森林 (默认参数) - 5折交叉验证平均AUC: {rf_default_score:.4f} ± {rf_default_std:.4f}")

# 贝叶斯超参数优化
rf_param_bounds = {
    'n_estimators': (10, 50),
    'max_depth': (1, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'max_features': (0.1, 0.5)
}

def rf_objective(**params):
    params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] else v 
            for k, v in params.items()}
    model = RandomForestClassifier(**params, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

rf_optimizer = BayesianOptimization(
    f=rf_objective,
    pbounds=rf_param_bounds,
    random_state=RANDOM_STATE
)

rf_optimizer.maximize(init_points=5, n_iter=25)
rf_best_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] else v 
                for k, v in rf_optimizer.max['params'].items()}
rf_best_params['random_state'] = RANDOM_STATE

print(f"最佳随机森林参数: {rf_best_params}")
print(f"随机森林 (调优参数) - 5折交叉验证平均AUC: {rf_optimizer.max['target']:.4f}")

# 训练最佳模型
rf_best = RandomForestClassifier(**rf_best_params)
rf_best.fit(X_train_selected, y_train)
models['RF'] = rf_best
model_performances['RF'] = {'default_auc': rf_default_score, 'optimized_auc': rf_optimizer.max['target'], 'params': rf_best_params}

# 4. 梯度提升 (GB)
print("\n4. 训练梯度提升模型...")
# 默认参数
gb_default = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb_default_score, gb_default_std = evaluate_model(gb_default, X_train_selected, y_train)
print(f"梯度提升 (默认参数) - 5折交叉验证平均AUC: {gb_default_score:.4f} ± {gb_default_std:.4f}")

# 贝叶斯超参数优化
gb_param_bounds = {
    'n_estimators': (10, 50),
    'learning_rate': (0.01, 0.3),
    'max_depth': (1, 10),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10),
    'subsample': (0.5, 1.0),
    'max_features': (0.1, 1.0)
}

def gb_objective(**params):
    params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] else v 
            for k, v in params.items()}
    model = GradientBoostingClassifier(**params, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

gb_optimizer = BayesianOptimization(
    f=gb_objective,
    pbounds=gb_param_bounds,
    random_state=RANDOM_STATE
)

gb_optimizer.maximize(init_points=5, n_iter=25)
gb_best_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'] else v 
                for k, v in gb_optimizer.max['params'].items()}
gb_best_params['random_state'] = RANDOM_STATE

print(f"最佳梯度提升参数: {gb_best_params}")
print(f"梯度提升 (调优参数) - 5折交叉验证平均AUC: {gb_optimizer.max['target']:.4f}")

# 训练最佳模型
gb_best = GradientBoostingClassifier(**gb_best_params)
gb_best.fit(X_train_selected, y_train)
models['GB'] = gb_best
model_performances['GB'] = {'default_auc': gb_default_score, 'optimized_auc': gb_optimizer.max['target'], 'params': gb_best_params}

# 5. XGBoost (XGB)
print("\n5. 训练XGBoost模型...")
# 默认参数
xgb_default = xgb.XGBClassifier(random_state=RANDOM_STATE)
xgb_default_score, xgb_default_std = evaluate_model(xgb_default, X_train_selected, y_train)
print(f"XGBoost (默认参数) - 5折交叉验证平均AUC: {xgb_default_score:.4f} ± {xgb_default_std:.4f}")

# 贝叶斯超参数优化
xgb_param_bounds = {
    'n_estimators': (10, 50),
    'learning_rate': (0.01, 0.3),
    'max_depth': (1, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'gamma': (0, 5),
    'min_child_weight': (1, 10)
}

def xgb_objective(**params):
    params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_child_weight'] else v 
            for k, v in params.items()}
    model = xgb.XGBClassifier(**params, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

xgb_optimizer = BayesianOptimization(
    f=xgb_objective,
    pbounds=xgb_param_bounds,
    random_state=RANDOM_STATE
)

xgb_optimizer.maximize(init_points=5, n_iter=25)
xgb_best_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_child_weight'] else v 
                for k, v in xgb_optimizer.max['params'].items()}
xgb_best_params['random_state'] = RANDOM_STATE

print(f"最佳XGBoost参数: {xgb_best_params}")
print(f"XGBoost (调优参数) - 5折交叉验证平均AUC: {xgb_optimizer.max['target']:.4f}")

# 训练最佳模型
xgb_best = xgb.XGBClassifier(**xgb_best_params)
xgb_best.fit(X_train_selected, y_train)
models['XGB'] = xgb_best
model_performances['XGB'] = {'default_auc': xgb_default_score, 'optimized_auc': xgb_optimizer.max['target'], 'params': xgb_best_params}

# 6. LightGBM (LGB)
print("\n6. 训练LightGBM模型...")
# 默认参数
lgb_default = lgb.LGBMClassifier(random_state=RANDOM_STATE)
lgb_default_score, lgb_default_std = evaluate_model(lgb_default, X_train_selected, y_train)
print(f"LightGBM (默认参数) - 5折交叉验证平均AUC: {lgb_default_score:.4f} ± {lgb_default_std:.4f}")

# 贝叶斯超参数优化
lgb_param_bounds = {
    'n_estimators': (10, 50),
    'learning_rate': (0.01, 0.3),
    'max_depth': (1, 10),
    'num_leaves': (2, 30),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'min_child_samples': (1, 50),
    'reg_alpha': (0, 1.0),
    'reg_lambda': (0, 1.0)
}

def lgb_objective(**params):
    params = {k: int(v) if k in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples'] else v 
            for k, v in params.items()}
    model = lgb.LGBMClassifier(**params, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

lgb_optimizer = BayesianOptimization(
    f=lgb_objective,
    pbounds=lgb_param_bounds,
    random_state=RANDOM_STATE
)

lgb_optimizer.maximize(init_points=5, n_iter=25)
lgb_best_params = {k: int(v) if k in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples'] else v 
                for k, v in lgb_optimizer.max['params'].items()}
lgb_best_params['random_state'] = RANDOM_STATE

print(f"最佳LightGBM参数: {lgb_best_params}")
print(f"LightGBM (调优参数) - 5折交叉验证平均AUC: {lgb_optimizer.max['target']:.4f}")

# 训练最佳模型
lgb_best = lgb.LGBMClassifier(**lgb_best_params)
lgb_best.fit(X_train_selected, y_train)
models['LGB'] = lgb_best
model_performances['LGB'] = {'default_auc': lgb_default_score, 'optimized_auc': lgb_optimizer.max['target'], 'params': lgb_best_params}

# 7. 支持向量机 (SVM) 
print("\n7. 训练SVM模型...")
# 默认参数
svm_default = SVC(random_state=RANDOM_STATE)
svm_default_score, svm_default_std = evaluate_model(svm_default, X_train_selected, y_train)
print(f"SVM(默认参数) - 5折交叉验证平均AUC: {svm_default_score:.4f} ± {svm_default_std:.4f}")

# 贝叶斯超参数优化
svm_param_bounds = {
    'C': (0.1, 50),
    'gamma': (0.001, 8.0)
}

def svm_objective(**params):
    model = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, **params)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

svm_optimizer = BayesianOptimization(
    f=svm_objective,
    pbounds=svm_param_bounds,
    random_state=RANDOM_STATE
)

svm_optimizer.maximize(init_points=5, n_iter=15)
svm_best_params = svm_optimizer.max['params']
svm_best_params['kernel'] = 'rbf'
svm_best_params['probability'] = True
svm_best_params['random_state'] = RANDOM_STATE

print(f"最佳SVM参数: {svm_best_params}")
print(f"SVM (调优参数) - 5折交叉验证平均AUC: {svm_optimizer.max['target']:.4f}")

# 训练最佳模型
svm_best = SVC(**svm_best_params)
svm_best.fit(X_train_selected, y_train)
models['SVM'] = svm_best
model_performances['SVM'] = {'default_auc': svm_default_score, 'optimized_auc': svm_optimizer.max['target'], 'params': svm_best_params}

# 8. 多层感知机 (MLP)
print("\n8. 训练MLP模型...")
# 默认参数
mlp_default = MLPClassifier(random_state=RANDOM_STATE)
mlp_default_score, mlp_default_std = evaluate_model(mlp_default, X_train_selected, y_train)
print(f"MLP (默认参数) - 5折交叉验证平均AUC: {mlp_default_score:.4f} ± {mlp_default_std:.4f}")

# 贝叶斯超参数优化
mlp_param_bounds = {
    'hidden_layer_sizes_1': (5, 50),
    'hidden_layer_sizes_2': (5, 50),
    'alpha': (0.0001, 0.1),
    'learning_rate_init': (0.001, 0.1)
}

def mlp_objective(**params):
    hidden_layer_sizes_1 = int(params.pop('hidden_layer_sizes_1'))
    hidden_layer_sizes_2 = int(params.pop('hidden_layer_sizes_2'))
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_sizes_1, hidden_layer_sizes_2),
        max_iter=1000,
        early_stopping=True,
        random_state=RANDOM_STATE,
        **params
    )
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

mlp_optimizer = BayesianOptimization(
    f=mlp_objective,
    pbounds=mlp_param_bounds,
    random_state=RANDOM_STATE
)

mlp_optimizer.maximize(init_points=5, n_iter=15)
mlp_best_params = mlp_optimizer.max['params']
mlp_best_params['hidden_layer_sizes'] = (int(mlp_best_params.pop('hidden_layer_sizes_1')), 
                                        int(mlp_best_params.pop('hidden_layer_sizes_2')))
mlp_best_params['max_iter'] = 1000
mlp_best_params['early_stopping'] = True
mlp_best_params['random_state'] = RANDOM_STATE

print(f"最佳MLP参数: {mlp_best_params}")
print(f"MLP (调优参数) - 5折交叉验证平均AUC: {mlp_optimizer.max['target']:.4f}")

# 训练最佳模型
mlp_best = MLPClassifier(**mlp_best_params)
mlp_best.fit(X_train_selected, y_train)
models['MLP'] = mlp_best
model_performances['MLP'] = {'default_auc': mlp_default_score, 'optimized_auc': mlp_optimizer.max['target'], 'params': mlp_best_params}

# 9. 朴素贝叶斯 (NB)
print("\n9. 训练朴素贝叶斯模型...")
# 默认参数
nb_default = GaussianNB()
nb_default_score, nb_default_std = evaluate_model(nb_default, X_train_selected, y_train)
print(f"朴素贝叶斯 (默认参数) - 5折交叉验证平均AUC: {nb_default_score:.4f} ± {nb_default_std:.4f}")

# 朴素贝叶斯没有太多可调整的参数，我们只调整var_smoothing
nb_param_bounds = {
    'var_smoothing': (1e-10, 1e-5)
}

def nb_objective(**params):
    model = GaussianNB(**params)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
    return cv_scores.mean()

nb_optimizer = BayesianOptimization(
    f=nb_objective,
    pbounds=nb_param_bounds,
    random_state=RANDOM_STATE
)

nb_optimizer.maximize(init_points=3, n_iter=10)
nb_best_params = nb_optimizer.max['params']

print(f"最佳朴素贝叶斯参数: {nb_best_params}")
print(f"朴素贝叶斯 (调优参数) - 5折交叉验证平均AUC: {nb_optimizer.max['target']:.4f}")

# 训练最佳模型
nb_best = GaussianNB(**nb_best_params)
nb_best.fit(X_train_selected, y_train)
models['NB'] = nb_best
model_performances['NB'] = {'default_auc': nb_default_score, 'optimized_auc': nb_optimizer.max['target'], 'params': nb_best_params}

# 10. K近邻 (KNN)
print("\n10. 训练KNN模型...")
# 默认参数
knn_default = KNeighborsClassifier()
knn_default_score, knn_default_std = evaluate_model(knn_default, X_train_selected, y_train)
print(f"KNN (默认参数) - 5折交叉验证平均AUC: {knn_default_score:.4f} ± {knn_default_std:.4f}")

# 贝叶斯超参数优化
knn_param_bounds = {
    'n_neighbors': (1, 10),
    'weights': ['uniform', 'distance'],
    'p': (1, 2)  # p=1 为曼哈顿距离，p=2 为欧几里得距离
}

# 针对weights的不同取值进行优化
knn_best_params = {}
knn_best_score = 0

for weights in ['uniform', 'distance']:
    param_bounds = {
        'n_neighbors': (1, 20),
        'p': (1, 2)
    }
    
    model_params = {
        'weights': weights
    }
    
    def objective(**params):
        params['n_neighbors'] = int(params['n_neighbors'])
        model = KNeighborsClassifier(**{**model_params, **params})
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='roc_auc')
        return cv_scores.mean()
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=param_bounds,
        random_state=RANDOM_STATE
    )
    
    optimizer.maximize(init_points=3, n_iter=10)
    
    if optimizer.max['target'] > knn_best_score:
        knn_best_score = optimizer.max['target']
        knn_best_params = {**model_params, **optimizer.max['params']}
        knn_best_params['n_neighbors'] = int(knn_best_params['n_neighbors'])

print(f"最佳KNN参数: {knn_best_params}")
print(f"KNN (调优参数) - 5折交叉验证平均AUC: {knn_best_score:.4f}")

# 训练最佳模型
knn_best = KNeighborsClassifier(**knn_best_params)
knn_best.fit(X_train_selected, y_train)
models['KNN'] = knn_best
model_performances['KNN'] = {'default_auc': knn_default_score, 'optimized_auc': knn_best_score, 'params': knn_best_params}

# 保存所有模型
for name, model in models.items():
    joblib.dump(model, f'D:/vscode_work/mechine learning/{name}_model.pkl')

# 保存模型性能信息
joblib.dump(model_performances, 'D:/vscode_work/mechine learning/model_performances.pkl')

print("\n所有模型训练完成并保存。")

#########################################4.模型评估#######################################
# ④ 模型评估
print("\n开始在测试集上评估模型...")

# 准备模型和颜色
model_names = list(models.keys())
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
model_colors = dict(zip(model_names, colors[:len(model_names)]))

# 计算置信区间的函数
def auc_ci(y_true, y_pred, alpha=0.95):
    auc_value = roc_auc_score(y_true, y_pred)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    q1 = auc_value / (2 - auc_value)
    q2 = 2 * auc_value**2 / (1 + auc_value)
    se = np.sqrt((auc_value * (1 - auc_value) + (n_pos - 1) * (q1 - auc_value**2) + 
                  (n_neg - 1) * (q2 - auc_value**2)) / (n_pos * n_neg))
    z = stats.norm.ppf(1 - (1 - alpha) / 2)
    lower = auc_value - z * se
    upper = auc_value + z * se
    return auc_value, lower, upper

# 定义决策曲线分析函数
def decision_curve_analysis(y_true, y_pred_proba, model_name, color):
    # 计算一系列阈值下的净收益
    thresholds = np.linspace(0, 1, 100)
    net_benefit = []
    
    for threshold in thresholds:
        # 根据阈值将概率转换为预测标签
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 计算真阳性和假阳性
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        # 计算净收益
        n = len(y_true)
        if tp + fp == 0:
            net_benefit.append(0)
        else:
            net_benefit.append((tp / n) - (fp / n) * (threshold / (1 - threshold)))
    
    return thresholds, net_benefit

# 创建结果存储字典
test_results = {}

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
for name, model in models.items():
    # 预测概率
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    # 计算预测标签
    y_pred = model.predict(X_test_selected)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 计算各种评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall = recall_score(y_test, y_pred)
    auc_value, auc_lower, auc_upper = auc_ci(y_test, y_pred_proba)
    
    # 存储结果
    test_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'sensitivity': sensitivity,
        'f1': f1,
        'specificity': specificity,
        'recall': recall,
        'auc': auc_value,
        'auc_lower': auc_lower,
        'auc_upper': auc_upper,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, lw=2, color=model_colors[name], 
            label=f'{name} (AUC = {auc_value:.3f}, 95% CI: [{auc_lower:.3f}-{auc_upper:.3f}])')

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('D:/vscode_work/mechine learning/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制Precision-Recall曲线
plt.figure(figsize=(10, 8))
for name in model_names:
    y_pred_proba = test_results[name]['y_pred_proba']
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    average_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.plot(recall_curve, precision_curve, lw=2, color=model_colors[name],
            label=f'{name} (AP = {average_precision:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves Comparison')
plt.legend(loc="best")
plt.grid(True)
plt.savefig('D:/vscode_work/mechine learning/pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制校准曲线
plt.figure(figsize=(10, 8))
for name in model_names:
    y_pred_proba = test_results[name]['y_pred_proba']
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', color=model_colors[name],
            label=f'{name}')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Probability')
plt.title('Calibration Curves Comparison')
plt.legend(loc="best")
plt.grid(True)
plt.savefig('D:/vscode_work/mechine learning/calibration_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 决策曲线分析
plt.figure(figsize=(10, 8))
net_benefits = {}

for name in model_names:
    y_pred_proba = test_results[name]['y_pred_proba']
    thresholds, net_benefit = decision_curve_analysis(y_test, y_pred_proba, name, model_colors[name])
    net_benefits[name] = net_benefit
    
    plt.plot(thresholds, net_benefit, lw=2, color=model_colors[name], label=name)

# 添加"全部干预"和"无干预"基准线
all_intervention = [np.sum(y_test) / len(y_test) - threshold / (1 - threshold) * (1 - np.sum(y_test) / len(y_test)) for threshold in thresholds]
plt.plot(thresholds, all_intervention, 'k--', lw=1.5, label='all intervention')
plt.plot(thresholds, [0] * len(thresholds), 'k-', lw=1.5, label='no intervention')

plt.xlim([0.0, 0.8])  # 通常只关注低阈值区域
plt.ylim([-0.05, max([max(net_benefits[name]) for name in model_names]) + 0.05])
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Effectiveness of Different Models')
plt.legend(loc="best")
plt.grid(True)
plt.savefig('D:/vscode_work/mechine learning/decision_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 使用DeLong检验比较AUC
from scipy import stats

def delong_test(y_true, y_pred_proba1, y_pred_proba2):
    """简化版DeLong检验，通过Z统计量近似p值"""
    # 获取阳性和阴性样本的索引
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    # 计算两组预测概率在阳性和阴性样本上的均值差异
    pos_diff = np.mean([y_pred_proba1[i] for i in pos_idx]) - np.mean([y_pred_proba2[i] for i in pos_idx])
    neg_diff = np.mean([y_pred_proba1[i] for i in neg_idx]) - np.mean([y_pred_proba2[i] for i in neg_idx])
    
    # 计算两组ROC曲线下面积的差异
    auc_diff = (pos_diff - neg_diff) / 2
    
    # 计算标准误差
    n1 = len(pos_idx)
    n2 = len(neg_idx)
    var1 = np.var([y_pred_proba1[i] for i in pos_idx]) / n1
    var2 = np.var([y_pred_proba2[i] for i in pos_idx]) / n1
    var3 = np.var([y_pred_proba1[i] for i in neg_idx]) / n2
    var4 = np.var([y_pred_proba2[i] for i in neg_idx]) / n2
    
    se = np.sqrt(var1 + var2 + var3 + var4)
    
    # 计算Z统计量
    z = auc_diff / se
    
    # 计算双侧p值
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return auc_diff, p

# 创建AUC对比表格
auc_comparison = np.zeros((len(model_names), len(model_names)))
p_values = np.zeros((len(model_names), len(model_names)))

for i, name1 in enumerate(model_names):
    for j, name2 in enumerate(model_names):
        if i == j:
            auc_comparison[i, j] = 0
            p_values[i, j] = 1
        else:
            auc_diff, p = delong_test(y_test, 
                                    test_results[name1]['y_pred_proba'], 
                                    test_results[name2]['y_pred_proba'])
            auc_comparison[i, j] = auc_diff
            p_values[i, j] = p

# 将结果转换为DataFrame并保存
auc_comparison_df = pd.DataFrame(auc_comparison, index=model_names, columns=model_names)
p_values_df = pd.DataFrame(p_values, index=model_names, columns=model_names)

auc_comparison_df.to_csv('D:/vscode_work/mechine learning/auc_comparison.csv')
p_values_df.to_csv('D:/vscode_work/mechine learning/p_values.csv')

# 打印模型评估结果表格
results_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [test_results[name]['accuracy'] for name in model_names],
    'Precision': [test_results[name]['precision'] for name in model_names],
    'Sensitivity': [test_results[name]['sensitivity'] for name in model_names],
    'Specificity': [test_results[name]['specificity'] for name in model_names],
    'F1 Score': [test_results[name]['f1'] for name in model_names],
    'AUC': [test_results[name]['auc'] for name in model_names],
    'AUC 95% CI': [f"[{test_results[name]['auc_lower']:.3f}-{test_results[name]['auc_upper']:.3f}]" for name in model_names]
})

print("\n测试集评估结果:")
print(results_df)
results_df.to_csv('D:/vscode_work/mechine learning/test_results.csv', index=False)

# 绘制混淆矩阵
for name in model_names:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_results[name]['y_pred'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'D:/vscode_work/mechine learning/{name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n模型评估完成。所有图表和结果已保存。")

#################################################5.外部验证###############################################

print("\n准备外部验证...")

# external_df = pd.read_csv('D:/vscode_work/mechine learning/外部验证插补log.csv')
# X_external = external_df.drop('outcome', axis=1)
# y_external = external_df['outcome']

# # 标准化外部验证集的数值特征
# X_external_scaled = X_external.copy()
# X_external_scaled[numeric_features] = scaler.transform(X_external[numeric_features])

# # 加载已保存的特征和模型
# final_selected_features = joblib.load('D:/vscode_work/mechine learning/final_selected_features.pkl')
# models = {}
# for name in ['LR', 'DT', 'RF', 'GB', 'XGB', 'LGB', 'SVM', 'MLP', 'NB', 'KNN']:
#     models[name] = joblib.load(f'D:\vscode_work\mechine learning{name}_model.pkl')

# 定义外部验证函数
def validate_external(X_external, y_external, models, final_selected_features):
    # 进行与测试集相同的预处理
    X_external_selected = X_external[final_selected_features]
    
    # 准备存储验证结果
    validation_results = {}
    
    # 评估所有模型
    for name, model in models.items():
        # 预测概率
        y_pred_proba = model.predict_proba(X_external_selected)[:, 1]
        # 计算预测标签
        y_pred = model.predict(X_external_selected)
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_external, y_pred)
        
        # 计算各种评估指标
        accuracy = accuracy_score(y_external, y_pred)
        precision = precision_score(y_external, y_pred)
        sensitivity = recall_score(y_external, y_pred)
        f1 = f1_score(y_external, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        recall = recall_score(y_external, y_pred)
        auc_value, auc_lower, auc_upper = auc_ci(y_external, y_pred_proba)
        
        # 存储结果
        validation_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'f1': f1,
            'specificity': specificity,
            'recall': recall,
            'auc': auc_value,
            'auc_lower': auc_lower,
            'auc_upper': auc_upper,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    for name in model_names:
        y_pred_proba = validation_results[name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_external, y_pred_proba)
        
        plt.plot(fpr, tpr, lw=2, color=model_colors[name], 
                label=f'{name} (AUC = {validation_results[name]["auc"]:.3f}, 95% CI: [{validation_results[name]["auc_lower"]:.3f}-{validation_results[name]["auc_upper"]:.3f}])')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('External Validation - ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('D:/vscode_work/mechine learning/external_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制Precision-Recall曲线
    plt.figure(figsize=(10, 8))
    for name in model_names:
        y_pred_proba = validation_results[name]['y_pred_proba']
        precision_curve, recall_curve, _ = precision_recall_curve(y_external, y_pred_proba)
        average_precision = average_precision_score(y_external, y_pred_proba)
        
        plt.plot(recall_curve, precision_curve, lw=2, color=model_colors[name],
                label=f'{name} (AP = {average_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('External Validation - Precision-Recall Curves Comparison')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('D:/vscode_work/mechine learning/external_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制校准曲线
    plt.figure(figsize=(10, 8))
    for name in model_names:
        y_pred_proba = validation_results[name]['y_pred_proba']
        fraction_of_positives, mean_predicted_value = calibration_curve(y_external, y_pred_proba, n_bins=10)
        
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', color=model_colors[name],
                label=f'{name}')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Probability')
    plt.title('External Validation - predictive Calibration Curves Comparison')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('D:/vscode_work/mechine learning/external_calibration_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 决策曲线分析
    plt.figure(figsize=(10, 8))
    net_benefits = {}
    
    for name in model_names:
        y_pred_proba = validation_results[name]['y_pred_proba']
        thresholds, net_benefit = decision_curve_analysis(y_external, y_pred_proba, name, model_colors[name])
        net_benefits[name] = net_benefit
        
        plt.plot(thresholds, net_benefit, lw=2, color=model_colors[name], label=name)
    
    # 添加"全部干预"和"无干预"基准线
    all_intervention = [np.sum(y_external) / len(y_external) - threshold / (1 - threshold) * (1 - np.sum(y_external) / len(y_external)) for threshold in thresholds]
    plt.plot(thresholds, all_intervention, 'k--', lw=1.5, label='all intervention')
    plt.plot(thresholds, [0] * len(thresholds), 'k-', lw=1.5, label='no intervention')
    
    plt.xlim([0.0, 0.8])
    plt.ylim([-0.05, max([max(net_benefits[name]) for name in model_names]) + 0.05])
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('External Validation - Effectiveness of Different Models')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig('D:/vscode_work/mechine learning/external_decision_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 使用DeLong检验比较AUC
    # 创建AUC对比表格
    auc_comparison = np.zeros((len(model_names), len(model_names)))
    p_values = np.zeros((len(model_names), len(model_names)))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                auc_comparison[i, j] = 0
                p_values[i, j] = 1
            else:
                auc_diff, p = delong_test(y_external, 
                                        validation_results[name1]['y_pred_proba'], 
                                        validation_results[name2]['y_pred_proba'])
                auc_comparison[i, j] = auc_diff
                p_values[i, j] = p
    
    # 将结果转换为DataFrame并保存
    auc_comparison_df = pd.DataFrame(auc_comparison, index=model_names, columns=model_names)
    p_values_df = pd.DataFrame(p_values, index=model_names, columns=model_names)
    
    auc_comparison_df.to_csv('D:/vscode_work/mechine learning/external_auc_comparison.csv')
    p_values_df.to_csv('D:/vscode_work/mechine learning/external_p_values.csv')
    
    # 打印模型评估结果表格
    results_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [validation_results[name]['accuracy'] for name in model_names],
        'Precision': [validation_results[name]['precision'] for name in model_names],
        'Sensitivity': [validation_results[name]['sensitivity'] for name in model_names],
        'Specificity': [validation_results[name]['specificity'] for name in model_names],
        'F1 Score': [validation_results[name]['f1'] for name in model_names],
        'AUC': [validation_results[name]['auc'] for name in model_names],
        'AUC 95% CI': [f"[{validation_results[name]['auc_lower']:.3f}-{validation_results[name]['auc_upper']:.3f}]" for name in model_names]
    })
    
    print("\n外部验证评估结果:")
    print(results_df)
    results_df.to_csv('D:/vscode_work/mechine learning/external_validation_results.csv', index=False)
    
    # 绘制混淆矩阵
    for name in model_names:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_external, validation_results[name]['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], 
                    yticklabels=['Negative', 'Positive'])
        
        plt.title(f'{name} External Validation Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'D:/vscode_work/mechine learning/{name}_external_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return validation_results


print("\n准备外部验证...")

def prepare_external_data(X_external, scaler, final_selected_features, categorical_features):
    # 获取 scaler fit 时的数值特征列（非常重要！）
    numeric_features_full = list(scaler.feature_names_in_)

    # Step 1: 补齐所有需要的数值型特征（以 scaler 为准）
    for col in numeric_features_full:
        if col not in X_external.columns:
            print(f"警告：外部验证集中缺少数值特征 '{col}'，使用训练集的均值进行填充。")
            train_mean = X_train[col].mean()
            X_external[col] = train_mean

    # Step 2: 补齐分类特征
    for col in categorical_features:
        if col not in X_external.columns:
            print(f"警告：外部验证集中缺少分类特征 '{col}'，使用默认值 0 进行填充。")
            X_external[col] = 0

    # Step 3: 确保列顺序和 scaler 要求一致
    X_external = X_external[numeric_features_full + categorical_features]

    # Step 4: 标准化
    X_external_scaled = X_external.copy()
    X_external_scaled[numeric_features_full] = scaler.transform(X_external_scaled[numeric_features_full])

    # Step 5: 筛选模型使用的特征
    X_external_selected = X_external_scaled[final_selected_features]

    return X_external_selected

# 加载数据
original_df = pd.read_csv('D:/vscode_work/mechine learning/data_im_e阴log.csv')
external_df = pd.read_csv('D:/vscode_work/mechine learning/外部验证插补log.csv')
X_external = external_df.drop('outcome', axis=1)
y_external = external_df['outcome']

# 加载特征和 scaler
final_selected_features = joblib.load('D:/vscode_work/mechine learning/final_selected_features.pkl')
scaler = joblib.load('D:/vscode_work/mechine learning/scaler.pkl')

# 获取分类特征（这部分是模型训练时就定好的）
categorical_features = ['sex', 'history', 'treatment', 'DNA']

# 准备外部验证数据
X_external_selected = prepare_external_data(X_external, scaler, final_selected_features, categorical_features)

# 检查是否每个选定的特征都在外部验证集中
missing_features = [f for f in final_selected_features if f not in X_external_selected.columns]
if missing_features:
    print(f"错误：外部验证集中缺少以下选定特征: {missing_features}")
else:
    # 加载模型
    models = {}
    import os
    for name in ['LR', 'DT', 'RF', 'GB', 'XGB', 'LGB', 'SVM', 'MLP', 'NB', 'KNN']:
        model_path = f'D:/vscode_work/mechine learning/{name}_model.pkl'
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)
            print(f"已加载模型: {name}")
        else:
            print(f"警告：找不到模型文件 {model_path}")

    # 执行外部验证
    external_validation_results = validate_external(X_external_selected, y_external, models, final_selected_features)

#################################################6.模型解释###############################################

# 1.导入库和数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import statsmodels.api as sm
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os

# 设置随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 加载数据和模型
X_train_selected, y_train, X_test_selected, y_test = joblib.load('D:/vscode_work/mechine learning/selected_train_test_data.pkl')
final_selected_features = joblib.load('D:/vscode_work/mechine learning/final_selected_features.pkl')

# 加载模型
lr_model = joblib.load('D:/vscode_work/mechine learning/LR_model.pkl')
mlp_model = joblib.load('D:/vscode_work/mechine learning/MLP_model.pkl')

print("数据和模型加载完成。")
print(f"训练集形状: {X_train_selected.shape}")
print(f"测试集形状: {X_test_selected.shape}")
print(f"选定特征数量: {len(final_selected_features)}")

# 2. 逻辑回归模型的SHAP解释

# 创建逻辑回归模型的SHAP解释器
print("生成逻辑回归模型的SHAP解释...")

# 创建SHAP解释器
explainer_lr = shap.LinearExplainer(lr_model, X_train_selected)

# 计算测试集的SHAP值
shap_values_lr = explainer_lr.shap_values(X_test_selected)

# 汇总图：显示每个特征的整体重要性
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_lr, X_test_selected, plot_type="bar", show=False)
plt.title("Logistic Regression - Feature Importance (SHAP value)", fontsize=15)
plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/LR_SHAP_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 详细摘要图：显示每个特征值对预测的影响
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_lr, X_test_selected, show=False)
plt.title("Logistic Regression - SHAP Values Summary Plot", fontsize=15)
plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/LR_SHAP_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# 依赖图：显示特定特征如何影响预测
# 选择SHAP值绝对值最大的前3个特征
feature_importance = np.abs(shap_values_lr).mean(0)
top_features_idx = np.argsort(feature_importance)[-3:]
top_features = [X_test_selected.columns[i] for i in top_features_idx]

for feature in top_features:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values_lr, X_test_selected, show=False)
    plt.title(f"Logistic Regression - {feature} dependence plot", fontsize=15)
    plt.tight_layout()
    plt.savefig(f'D:/vscode_work/mechine learning/LR_SHAP_dependence_{feature}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("逻辑回归SHAP解释完成。")

# 3. 逻辑回归列线图
# 创建逻辑回归模型的列线图
print("生成逻辑回归模型的列线图...")

def create_nomogram(model, X, feature_names, output_path):
    """
    为逻辑回归模型创建列线图
    
    参数:
    model - 已训练的逻辑回归模型
    X - 特征数据框
    feature_names - 特征名称列表
    output_path - 输出图像的路径
    """
    # 获取模型系数和截距
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    
    # 计算每个特征的影响范围
    min_max_diff = {}
    for i, feature in enumerate(feature_names):
        feature_min = X[feature].min()
        feature_max = X[feature].max()
        coef = coefs[i]
        min_max_diff[feature] = abs(coef * (feature_max - feature_min))
    
    # 按影响大小排序特征
    sorted_features = sorted(min_max_diff.items(), key=lambda x: x[1], reverse=True)
    
    # 设置图形
    fig = plt.figure(figsize=(12, len(feature_names) * 0.5 + 4))
    gs = gridspec.GridSpec(len(feature_names) + 2, 1)
    
    # 绘制特征轴
    ax_features = []
    max_points = 100  # 用于标准化
    
    # 绘制总分轴（顶部）
    ax_total = plt.subplot(gs[0, 0])
    ax_total.set_xlim(0, max_points)
    ax_total.set_xticks(np.linspace(0, max_points, 6))
    ax_total.set_xticklabels([f"{x:.0f}" for x in np.linspace(0, max_points, 6)])
    ax_total.set_title("total points", fontsize=12)
    ax_total.set_yticks([])
    ax_total.spines['left'].set_visible(False)
    ax_total.spines['right'].set_visible(False)
    ax_total.spines['top'].set_visible(False)
    
    # 绘制每个特征轴
    for i, (feature, diff) in enumerate(sorted_features):
        ax = plt.subplot(gs[i + 1, 0])
        ax_features.append(ax)
        
        # 获取特征的范围
        feature_min = X[feature].min()
        feature_max = X[feature].max()
        
        # 设置刻度
        feature_range = np.linspace(feature_min, feature_max, 5)
        
        # 计算特征值对应的分数
        idx = feature_names.index(feature)
        coef = coefs[idx]
        
        # 缩放系数使其适合0-100的范围
        max_abs_coef = max(abs(c) for c in coefs)
        scaled_coef = coef / max_abs_coef * (max_points / 2)
        
        # 计算每个特征值对应的分数
        points = [(val - feature_min) * scaled_coef / (feature_max - feature_min) + (max_points / 2 if coef < 0 else 0) for val in feature_range]
        
        # 绘制轴
        ax.set_xlim(min(points), max(points))
        ax.set_xticks(points)
        ax.set_xticklabels([f"{val:.2f}" for val in feature_range])
        ax.set_ylabel(feature, fontsize=10, rotation=0, ha='right', va='center')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    # 绘制概率轴（底部）
    ax_prob = plt.subplot(gs[-1, 0])
    
    # 计算不同分数对应的概率
    logits = np.linspace(-5, 5, 100)
    probs = 1 / (1 + np.exp(-logits))
    
    # 缩放至0-100分
    scaled_logits = (logits + 5) * (max_points / 10)
    
    ax_prob.set_xlim(0, max_points)
    ax_prob.set_xticks(np.linspace(0, max_points, 11))
    
    # 概率标签
    prob_labels = [f"{prob:.1%}" for prob in np.linspace(0, 1, 11)]
    ax_prob.set_xticklabels(prob_labels)
    ax_prob.set_title("Probability", fontsize=12)
    ax_prob.set_yticks([])
    ax_prob.spines['left'].set_visible(False)
    ax_prob.spines['right'].set_visible(False)
    ax_prob.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# 创建列线图
create_nomogram(
    lr_model, 
    X_train_selected, 
    final_selected_features, 
    'D:/vscode_work/mechine learning/LR_nomogram.png'
)

print("列线图生成完成。")

# 4. MLP模型SHAP解释
# 创建MLP模型的SHAP解释器
print("生成MLP模型的SHAP解释...")

# 由于MLP是黑盒模型，使用KernelExplainer
# 使用训练集的子集作为背景数据
background = shap.kmeans(X_train_selected, 50)  # 使用kmeans选择代表性样本
explainer_mlp = shap.KernelExplainer(mlp_model.predict_proba, background)

# 为了提高计算效率，可以使用测试集的子集
n_samples = min(100, X_test_selected.shape[0])  # 最多使用100个样本
X_test_sample = X_test_selected.sample(n_samples, random_state=RANDOM_STATE)

# 计算SHAP值 (注意：这一步可能需要一些时间)
shap_values_mlp = explainer_mlp.shap_values(X_test_sample)
# MLP返回每个类的SHAP值，我们使用阳性类（索引1）
shap_values_mlp_class1 = shap_values_mlp[1]

# 汇总图：显示每个特征的整体重要性
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_mlp[:,:,1], X_test_sample, plot_type="bar", show=False)
plt.title("MLP Model - Features Importance (SHAP值)", fontsize=15)
plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/MLP_SHAP_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 详细摘要图：显示每个特征值对预测的影响
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values_mlp[:,:,1], X_test_sample, show=False)
plt.title("MLP Model - Summary Plot", fontsize=15)
plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/MLP_SHAP_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# 依赖图：显示特定特征如何影响预测
plt.figure(figsize=(10, 6))
shap.dependence_plot("HBsAg12w", shap_values_mlp[:,:,1], X_test_sample, interaction_index="HBsAg", show=False)
plt.title("MLP Model - Dependence Plot", fontsize=15)
plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/MLP_SHAP_dependence.png', dpi=300, bbox_inches='tight')
plt.close()

# 单变量依赖图
plt.figure(figsize=(10, 6))
shap.dependence_plot("HBsAg12w", shap_values_mlp[:,:,1], X_test_sample, interaction_index=None, show=False)
plt.title("MLP Model - Dependence Plot", fontsize=15)
plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/MLP_SHAP_single_dependence.png', dpi=300, bbox_inches='tight')
plt.close()

# 力图
example_index = 0  # Explain the first sample in the test set
plt.figure(figsize=(24, 10))

# Force plot for MLP model - use the correct indexing and structure
shap.force_plot(
    explainer_mlp.expected_value[1],  # Base value for positive class
    shap_values_mlp_class1[example_index, :],  # SHAP values for a single example
    X_test_sample.iloc[example_index, :],  # Feature values for that example
    matplotlib=True,
    show=False
)
plt.figtext(0.5, 0.01, "MLP Model - Force Plot (Example #" + str(example_index) + ")", 
            fontsize=15, ha='center', va='bottom')
# plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/MLP_SHAP_force.png', dpi=300, bbox_inches='tight')
plt.close()

# 瀑布图
plt.figure(figsize=(12, 8))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values_mlp_class1[example_index, :],
        base_values=explainer_mlp.expected_value[1],
        data=X_test_sample.iloc[example_index, :].values,
        feature_names=X_test_sample.columns
    ),
    show=False
)
#plt.title("MLP Model - Waterfall Plot (Example #" + str(example_index) + ")", fontsize=15)
plt.figtext(0.5, 0.01, "MLP Model - Waterfall Plot (Example #" + str(example_index) + ")", 
            fontsize=15, ha='center', va='bottom')
plt.tight_layout()
plt.savefig('D:/vscode_work/mechine learning/MLP_SHAP_waterfall.png', dpi=300, bbox_inches='tight')
plt.close()

print("MLP SHAP解释完成。")