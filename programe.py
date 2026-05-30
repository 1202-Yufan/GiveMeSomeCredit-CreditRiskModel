#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBClassifier, plot_importance
#%%
#读数据
df=pd.read_csv('cs-training.csv')
df

#%%
#删除重复值和ID列
df=df.drop_duplicates()
df=df.drop(columns=['Unnamed: 0'])
df

# %%
#观察数据信息
df.isnull().sum()
# %%
df.info()
# %%
df.describe()


# %%
#发现年龄又有异常值，我们只看年龄大于18小于85的
df=df[(df['age']>=18)&(df['age']<=85)]
df.describe()



# %%
#发现三个numberoftime有异常值，变为nan
print(df['NumberOfTime30-59DaysPastDueNotWorse'].unique(),'\n')
print(df['NumberOfTime60-89DaysPastDueNotWorse'].unique(),'\n')
print(df['NumberOfTimes90DaysLate'].unique(),'\n')
# %%
outliner=[98,96]
df['NumberOfTime30-59DaysPastDueNotWorse']=df['NumberOfTime30-59DaysPastDueNotWorse'].replace(outliner,np.nan)
df['NumberOfTime60-89DaysPastDueNotWorse']=df['NumberOfTime60-89DaysPastDueNotWorse'].replace(outliner,np.nan)
df['NumberOfTimes90DaysLate']=df['NumberOfTimes90DaysLate'].replace(outliner,np.nan)
df.describe()




# %%
#发现RevolvingUtilizationOfUnsecuredLines和DebtRatio和MonthlyIncome有极大值，用盖帽法
up_rev=df['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99)
up_deb=df['DebtRatio'].quantile(0.99)
up_mon=df['MonthlyIncome'].quantile(0.99)
df.loc[df['RevolvingUtilizationOfUnsecuredLines']>=up_rev,'RevolvingUtilizationOfUnsecuredLines']=up_rev
df.loc[df['DebtRatio']>=up_deb,'DebtRatio']=up_deb
df.loc[df['MonthlyIncome']>=up_mon,'MonthlyIncome']=up_mon
df.describe()

# %%
#画his图和box图
select_index=df.select_dtypes(include=['float64','int64']).columns
n_col=4
n_row=math.ceil(len(select_index)/n_col)
plt.figure(figsize=(29,20))
for i,col in enumerate(select_index):
    plt.subplot(n_row,n_col,i+1)
    sns.histplot(x=df[col],kde= True,color='red')
    plt.title(f'{col} Distribution')
plt.tight_layout()
plt.show()
# %%
plt.figure(figsize=(39,30))
for i,col in enumerate(select_index):
    plt.subplot(n_row,n_col,i+1)
    sns.boxplot(y=df[col],color='skyblue')
    plt.title(f'{col} Outliers')
plt.tight_layout()
plt.show()



# %%
#画热力图看共线性
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()
#结果发现没有共线性



#线性
# %%
df_lin=df.copy()
df_lin.describe()
df_lin.isna().sum()
#%%
#1. 先把那些填固定 0 的处理掉 (因为 0 不涉及泄露)
cols_fill_zero = ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 
                  'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']
df_lin[cols_fill_zero] = df_lin[cols_fill_zero].fillna(0)
df_lin.isna().sum()
#%%
#分训练集，对于线性的
y=df_lin['SeriousDlqin2yrs']
X=df_lin.drop(columns=['SeriousDlqin2yrs'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
#%%
#补充空白值
X_train['MonthlyIncome']=X_train['MonthlyIncome'].fillna(X_train['MonthlyIncome'].median())
X_val['MonthlyIncome']=X_val['MonthlyIncome'].fillna(X_train['MonthlyIncome'].median())
# 1. 对 训练集 (X_train) 进行 Log 转换
X_train['MonthlyIncome_Log'] = np.log1p(X_train['MonthlyIncome'])

# 2. 对 测试/验证集 (X_val) 进行同样的 Log 转换
# 注意：一定要做！不然模型在训练时看的是 Log 后的数据，考试时看原始数据就乱套了
X_val['MonthlyIncome_Log'] = np.log1p(X_val['MonthlyIncome'])



# %%
#利用双变量分析”（Bivariate Analysi）查看分箱,要在分了训练集后，防止数据泄漏
df_train = X_train.copy()
df_train['SeriousDlqin2yrs'] = y_train
target='SeriousDlqin2yrs'
features=df_train.drop(columns=[target]).select_dtypes(include=['float64','int64']).columns
plt.figure(figsize=(15, 12))

for i, col in enumerate(features):
        plt.subplot(4, 3, i+1)
        
        # --- 策略 A: 如果是“少值特征” (比如 Overdue 只有5个值) ---
        # 我们画“违约率图”：看逾期次数越多，坏人比例是不是越高？
        if df_train[col].nunique() < 10:
            # 计算每个值的违约率 (Mean of Target)
            sns.barplot(x=col, y=target, data=df_train, errorbar=None, palette='Reds')
            plt.title(f'{col}: Default Rate (Higher is Worse)', color='red')
            plt.ylabel('Probability of Default')
            
        # --- 策略 B: 如果是“连续特征” (比如 Age, DebtRatio) ---
        # 我们画“箱线图”：对比好人(0)和坏人(1)的分布差异
        #看 target=0 和 target=1 的两个箱子，
        # 在 y 轴方向上是否明显分开。
        #明显有区分不再同一水平线就说明这个x要分箱，反之就不用
        else:
            sns.boxplot(x=target, y=col, data=df_train, showfliers=False, palette='Set2')
            plt.title(f'{col} Distribution by Target', color='blue')
            # 0是好人，1是坏人
    
plt.tight_layout()
plt.show()
# %%
#分箱子和#特征工程，转化为 One-Hot Encoding
# 1. 定义一个处理函数 (把刚才所有的分箱逻辑都放进去)
def process_binning(df_input):
    # 为了不修改原始数据，创建一个副本
    df = df_input.copy()
    # age分箱子
    df['Age']=pd.cut(df['age'],bins=[18,41,52,62,86],labels=['young','medium young','medium old','old'])
    # 信用卡额度用分箱子
    df['Rev']=pd.cut(df['RevolvingUtilizationOfUnsecuredLines'],bins=[-1,0.05,0.16,0.6,1.1],labels=['goodre','mediumgoodre','mediumbadre','badre'])

    # 30-59天逾期：切分成 [0次], [1次], [2次], [3次及以上]
    df['30-59Days_Bin'] = pd.cut(df['NumberOfTime30-59DaysPastDueNotWorse'], bins=[-1, 0, 1, 2, 100], labels=['Never_30_59', 'Once_30_59', 'Twice_30_59', 'More_30_59'])
    # 60-89天逾期：切分成 [0次], [1次], [2次及以上]
    df['60-89Days_Bin'] = pd.cut(df['NumberOfTime60-89DaysPastDueNotWorse'], bins=[-1, 0, 1, 100], labels=['Never_60_89', 'Once_60_89', 'More_60_89'])

    # 90天+逾期：切分成 [0次], [1次], [2次及以上]
    df['90Days_Bin'] = pd.cut(df['NumberOfTimes90DaysLate'], bins=[-1, 0, 1, 100], labels=['Never_90', 'Once_90', 'More_90'])

    # 针对家属数量分箱：0, 1, 2, 3+
    # bins=[-1, 0, 1, 2, 100] 的逻辑是：
    # (-1, 0] -> 0
    # (0, 1]  -> 1
    # (1, 2]  -> 2
    # (2, 100]-> 3到100 (即3个及以上)

    df['Dependents_Bin'] = pd.cut(df['NumberOfDependents'], bins=[-1, 0, 1, 2, 100], labels=['No_Dep', 'One_Dep', 'Two_Dep', 'More_Dep'])
    df=pd.get_dummies(df,columns=['Age','Rev','30-59Days_Bin','60-89Days_Bin','90Days_Bin','Dependents_Bin'],drop_first=True)
    df=df.drop(columns=['age','RevolvingUtilizationOfUnsecuredLines','NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime60-89DaysPastDueNotWorse','MonthlyIncome'])
    return df

X_train_lin = process_binning(X_train)  # 处理训练集
X_val_lin   = process_binning(X_val)    # 处理验证集
#%%
#假设 X_train 里有“90天逾期_3次以上”的人，但是 X_val（验证集）里恰好没有这个人（大家都很乖）。 
#后果： pd.get_dummies 生成后，训练集会有一列 90Days_Bin_More_90，但验证集里就没有这一列。 结局： 运行 model.predict(X_val) 时直接报错："ValueError: shapes do not match"（列数不对）。

#解决办法： 使用 Pandas 的 align 功能，强制让两边的列保持一致，缺少的自动补 0
# 关键步骤 3：列对齐 (防止 validation 集缺某一列报错)
# =======================================================
X_train_lin, X_val_lin = X_train_lin.align(X_val_lin, join='outer', axis=1, fill_value=0)
# %%
# 不要把“0/1哑变量”也归一化了
# 你的代码里 scaler.fit_transform(X_train_lin) 是把整个表都拿去标准化了。

# 连续变量（如负债率、收入Log）： 需要标准化，变成均值为0、方差为1的数值。

# 0/1 哑变量（如 Age_young）： 本来就是 0 和 1，如果强行标准化，会变成 -0.45 和 2.33 这样的奇怪小数。
# 影响： 虽然模型也能跑，但会失去“是否为年轻人”这种直观的可解释性。
#只归一化连续变量 (Best Practice)
# =======================================================
from sklearn.preprocessing import StandardScaler
# 挑选出还没变哑变量的连续列
cols_to_scale = ['DebtRatio', 'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines', 'MonthlyIncome_Log']

scaler = StandardScaler()

# 只在这些列上做计算
X_train_lin[cols_to_scale] = scaler.fit_transform(X_train_lin[cols_to_scale])
X_val_lin[cols_to_scale]   = scaler.transform(X_val_lin[cols_to_scale])
# %%
#训练模型：创建并训练逻辑回归模型。
from sklearn.model_selection import GridSearchCV
#这个是进行预测的，但是算信用分的话，这个因为有lasso/ridge所以这个违约概率是经过改变的，不是真实概率
#因此我们这个最后算评分卡时要加一步校准，讲这个数学概率转换为真实概率在感觉公式算其信用分数，不然回出现决策失误
logmodel=LogisticRegression(solver='liblinear',class_weight='balanced')

param={'C':[0.1,1,10,100],'penalty':['l1','l2']}
grid_search=GridSearchCV(logmodel,param,cv=5,scoring='roc_auc')
grid_search.fit(X_train_lin,y_train)
final_model=grid_search.best_estimator_
# %%
#查看结果
y_pred=final_model.predict(X_val_lin)
y_pro=final_model.predict_proba(X_val_lin)[:,1]
accu=accuracy_score(y_val,y_pred)
print("准确率:", accu,'\n')
conf_matrix=confusion_matrix(y_val,y_pred)
print('混淆矩阵：',conf_matrix,'\n')
class_report=classification_report(y_val,y_pred)
print("分类报告：",class_report)
auc_value = roc_auc_score(y_val, y_pro)
print(f"模型 AUC 值: {auc_value:.4f}")
# %%
#画roc_auc图
frp,trp,thresholds=roc_curve(y_val,y_pro)
plt.figure(figsize=(6,4))
plt.plot(frp,trp,label=f'AUC={auc_value:.2f}')
plt.plot([0,1],[0,1],'r--',label='Random Guess')
plt.xlabel('False Positive rate')
plt.ylabel('True positive rate')
plt.title('logitreg roc curve')
plt.legend()
plt.tight_layout()
plt.show()





# %%
#非线性

df_nonlin=df.copy()
df_nonlin.describe()
df_nonlin.isna().sum()
#%%
#1. 先把那些填固定 0 的处理掉 (因为 0 不涉及泄露)
cols_fill_zero = ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 
                  'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']
df_nonlin[cols_fill_zero] = df_nonlin[cols_fill_zero].fillna(0)
df_nonlin.isna().sum()
#%%
#分训练集，对于非线性的
yn=df_nonlin['SeriousDlqin2yrs']
Xn=df_nonlin.drop(columns=['SeriousDlqin2yrs'])
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(Xn, yn, test_size=0.2, random_state=42,stratify=yn)
#%%
#补充空白值
X_train_tree['MonthlyIncome']=X_train_tree['MonthlyIncome'].fillna(X_train_tree['MonthlyIncome'].median())
X_test_tree['MonthlyIncome']=X_test_tree['MonthlyIncome'].fillna(X_train_tree['MonthlyIncome'].median())






# %%
#随机森林
#训练找参数最优
rf_model=RandomForestClassifier(class_weight='balanced',random_state=42,n_jobs=-1)
param_rf={'n_estimators':randint(50,300),
          'max_depth':randint(5,16),
          "max_features": ["sqrt", "log2"],
          "min_samples_leaf": randint(150, 200),
          "min_samples_split": randint(2, 30),
          "max_leaf_nodes": randint(10, 60),
          "bootstrap": [True]}
rf_serach=RandomizedSearchCV(rf_model,param_rf,n_iter=50,cv=5,scoring='roc_auc',random_state=42,n_jobs=-1)
rf_serach.fit(X_train_tree,y_train_tree)
# %%
#检查是否过拟合
best_modelrn=rf_serach.best_estimator_
y_pred_rf=best_modelrn.predict(X_test_tree)
y_prob_rf=best_modelrn.predict_proba(X_test_tree)[:,1]
y_train_prob_rf=best_modelrn.predict_proba(X_train_tree)[:,1]
auc_train_rf=roc_auc_score(y_train_tree,y_train_prob_rf)
print(f"\nTrain AUC Score: {auc_train_rf:.4f}")

auc_rf = roc_auc_score(y_test_tree, y_prob_rf)
print(f"\nTest AUC Score: {auc_rf:.4f}")

#%%
# 打印详细报告
print("\n分类报告:")
print(classification_report(y_test_tree, y_pred_rf))
# %%
#画rf auc图
frp,trp,thresholds=roc_curve(y_test_tree,y_prob_rf)
plt.figure(figsize=(6,4))
plt.plot(frp,trp,label=f'rf AUC={auc_rf:.2f}')
plt.plot([0,1],[0,1],'k--',label='Random Guess')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('rf AUC Curve')
plt.legend()
plt.tight_layout()
plt.show()
# %%
#可视化：特征重要性 (Feature Importance)
# ==========================================
# 随机森林没法画出每一棵树（太多了），
# 但我们可以看：谁在委员会里说话分量最重？

importances = best_modelrn.feature_importances_
indices = np.argsort(importances)[::-1] # 排序


plt.figure(figsize=(15,10))
plt.title("Feature Importance (What the Forest cares about)")
plt.bar(range(Xn.shape[1]), importances[indices], align="center", color='skyblue')
plt.xticks(range(Xn.shape[1]), [Xn.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()


# %%
#xgboost
#训练找参数最优
neg=(y_train_tree==0).sum()
pos=(y_train_tree==1).sum()
scaler_weight=neg/pos
xgb_model=XGBClassifier(objective='binary:logistic',
                        random_state=42,n_jobs=-1, 
                        tree_method='hist')
param_xgb={
    # --- 核心 1: 学习率 (步子迈多大) ---
    # 越小越稳，但需要更多的树 (n_estimators 就要变大)
    "learning_rate": uniform(0.01, 0.05), 
    "n_estimators": randint(300, 600) ,          # 树多一点来弥补低学习率
    # --- 核心 2: 树的结构 ---
    "max_depth": randint(3, 6),        # XGB不建议太深，3-6通常够了
    "min_child_weight": randint(50, 80), # 必须凑够权重才准切分,
    #就是决策树/随机森林的min_samples_leaf (按人数算),限制叶子最小样本

    # --- 核心 3: 不平衡处理 ---
    # 我们允许在算出来的权重基础上上下浮动，太大容易欠拟合
    "scale_pos_weight": [1, np.sqrt(scaler_weight),scaler_weight],
    
    # --- 核心 4: 正则化 (防过拟合) ---
    # 这是 Random Forest 没有的，XGBoost 的独门绝技
    "reg_alpha": uniform(5, 10),  # L1 正则 (Lasso)
    "reg_lambda": uniform(5, 10), # L2 正则 (Ridge)
    "gamma": uniform(0.6, 1.5),
    
    # --- 核心 5: 随机抽样 (防过拟合) ---
    "subsample": uniform(0.5, 0.3),      # 每棵树只看 50%-80% 的数据
    "colsample_bytree": uniform(0.5, 0.3)# 每棵树只看 50%-80% 的特征
}
xgb_search=RandomizedSearchCV(xgb_model,param_xgb,n_iter=50,cv=5,random_state=42,n_jobs=-1,scoring='roc_auc',verbose=1)
xgb_search.fit(X_train_tree,y_train_tree)
# %%
best_modelxgb=xgb_search.best_estimator_
y_pred_xgb=best_modelxgb.predict(X_test_tree)
y_prob_xgb=best_modelxgb.predict_proba(X_test_tree)[:,1]
y_train_prob_xgb=best_modelxgb.predict_proba(X_train_tree)[:,1]
auc_train_xgb=roc_auc_score(y_train_tree,y_train_prob_xgb)
print(f"\nTrain AUC Score: {auc_train_xgb:.4f}")

auc_xgb = roc_auc_score(y_test_tree, y_prob_xgb)
print(f"\nTest AUC Score: {auc_xgb:.4f}")
# %%
#画出xgb AUC图
frp,trp,thresholds=roc_curve(y_test_tree,y_prob_xgb)
plt.figure(figsize=(6,4))
plt.plot(frp,trp,label=f'xgb AUC={auc_xgb:.2f}')
plt.plot([0,1],[0,1],'k--',label='Random Guess')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('xgb AUC Curve')
plt.legend()
plt.tight_layout()
plt.show()
# %%
# ============================================
# 7. 可视化：XGBoost 自带的特征重要性
# ============================================
# XGBoost 有三种计算重要性的方法：
# 1. weight (默认): 这个特征在所有树里被用来切分的次数 (Feature Split)
# 2. gain (推荐): 这个特征带来的平均增益 (用了它，Loss下降了多少?)
# 3. cover: 这个特征覆盖了多少样本

plt.figure(figsize=(10, 5))
# 这里我们画 'gain'，因为它通常更能反映特征的真实价值
plot_importance(best_modelxgb, importance_type='gain', max_num_features=10, height=0.5, color='purple')
plt.title("XGBoost Feature Importance (Gain)")
plt.show()




# %%
#三个AUC图画一起对比
plt.figure(figsize=(8,6))
frp_lg,trp_lg,thresholds=roc_curve(y_val,y_pro)
frp_rf,trp_rf,thresholds=roc_curve(y_test_tree,y_prob_rf)
frp_xgb,trp_xgb,thresholds=roc_curve(y_test_tree,y_prob_xgb)
plt.plot(frp_lg,trp_lg,label=f'log AUC={auc_value:.2f}',color='red')
plt.plot(frp_rf,trp_rf,label=f'rf AUC={auc_rf:.2f}',color='blue')
plt.plot(frp_xgb,trp_xgb,label=f'xgb AUC={auc_xgb:.2f}',color='green')
plt.plot([0,1],[0,1],'k--',label='Random Guess')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Model Comparison: ROC Curves')
plt.legend()
plt.tight_layout()
plt.show()
# %%
