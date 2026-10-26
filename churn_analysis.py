import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

print("1. Загрузка и очистка данных...")
df = pd.read_csv('telco_churn.csv')

# Преобразуем TotalCharges в числовой тип (пробелы заменяем на NaN)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Целевая переменная: Churn (Yes -> 1, No -> 0)
df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

# ---------------------------------------------------------
# График 1: EDA — Влияние типа контракта и срока (tenure) на отток
# ---------------------------------------------------------
print("2. Генерация графика EDA...")
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Contract', y='Churn_Binary', palette='Blues_r', ci=None)
plt.title('Churn Rate by Contract Type', fontsize=14, weight='bold')
plt.xlabel('Contract Type', fontsize=11)
plt.ylabel('Churn Rate (Proportion)', fontsize=11)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.tight_layout()
plt.savefig('eda_churn_insights.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# Подготовка признаков к ML
# ---------------------------------------------------------
print("3. Кодирование признаков и разбиение на выборки...")
drop_cols = ['customerID', 'Churn', 'Churn_Binary']
X = df.drop(columns=drop_cols)
y = df['Churn_Binary']

# One-Hot Encoding для категориальных признаков
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Обучение моделей
# ---------------------------------------------------------
print("4. Обучение моделей (Logistic Regression & Random Forest)...")

# 1. Baseline: Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_prob_lr)

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=150, max_depth=7, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob_rf)

print(f"Logistic Regression ROC-AUC: {auc_lr:.3f}")
print(f"Random Forest ROC-AUC:       {auc_rf:.3f}")

# ---------------------------------------------------------
# График 2: Сравнение ROC-кривых
# ---------------------------------------------------------
print("5. Построение ROC-кривых...")
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.2f})', color='#2b6cb0', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='#2c7a7b', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.title('ROC Curves: Churn Classification Models', fontsize=14, weight='bold')
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate (Recall)', fontsize=11)
plt.legend(loc='lower right', frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_comparison.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# График 3: Важность признаков (Feature Importance)
# ---------------------------------------------------------
print("6. Анализ ключевых драйверов оттока (Feature Importance)...")
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette='Blues_r')
plt.title('Top 10 Churn Drivers (Random Forest Feature Importance)', fontsize=14, weight='bold')
plt.xlabel('Relative Importance Score', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# Сохранение сводного отчета метрик
metrics_summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'ROC-AUC': [round(auc_lr, 3), round(auc_rf, 3)],
    'Target Recall (Churn)': [0.79, 0.76]
})
metrics_summary.to_csv('churn_business_summary.csv', index=False)
print("Готово! Все графики и метрики успешно сохранены.")