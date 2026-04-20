import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# ─── Paleta de colores ────────────────────────────────────────────────────────
C1, C2, C3 = '#4C72B0', '#DD8452', '#55A868'
GRAY = '#AAAAAA'
BG = '#F8F9FA'

# ─── 1. CARGA DE DATOS ───────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/dataset_estudiantes.csv')
print("="*60)
print("1. CARGA DE DATOS")
print("="*60)
print(f"Shape: {df.shape}")
print(df.dtypes)
print()

# ─── 2. ANÁLISIS EXPLORATORIO ────────────────────────────────────────────────
print("="*60)
print("2. ANÁLISIS EXPLORATORIO")
print("="*60)
print("\nEstadísticas descriptivas (numéricas):")
print(df.describe().round(2))
print("\nValores nulos por columna:")
print(df.isnull().sum())
print(f"\nRegistros duplicados: {df.duplicated().sum()}")
print(f"\nDistribución 'aprobado': {df['aprobado'].value_counts().to_dict()}")

# Figura 1: EDA — distribuciones y correlaciones
fig = plt.figure(figsize=(18, 14), facecolor=BG)
fig.suptitle('Análisis Exploratorio de Datos – Dataset Estudiantes', fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

num_cols = ['horas_estudio_semanal', 'nota_anterior', 'tasa_asistencia', 'horas_sueno', 'nota_final']

# Histogramas numéricas
for i, col in enumerate(num_cols):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.hist(df[col].dropna(), bins=30, color=C1, edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_', ' ').title(), fontsize=10, fontweight='bold')
    ax.set_xlabel('Valor', fontsize=8)
    ax.set_ylabel('Frecuencia', fontsize=8)
    ax.set_facecolor(BG)
    ax.grid(axis='y', alpha=0.3)

# Mapa de correlación
ax_corr = fig.add_subplot(gs[1:, 2])
num_df = df.select_dtypes(include='number')
corr = num_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax_corr, square=True, linewidths=0.5,
            annot_kws={'size': 7})
ax_corr.set_title('Correlación entre variables', fontsize=10, fontweight='bold')
ax_corr.tick_params(axis='x', labelsize=7, rotation=45)
ax_corr.tick_params(axis='y', labelsize=7)

# Aprobados vs suspendidos
ax_ap = fig.add_subplot(gs[2, 0])
counts = df['aprobado'].value_counts()
ax_ap.bar(['Suspendido (0)', 'Aprobado (1)'], [counts.get(0, 0), counts.get(1, 0)],
           color=[C2, C3], edgecolor='white')
ax_ap.set_title('Distribución Aprobado', fontsize=10, fontweight='bold')
ax_ap.set_ylabel('Nº alumnos', fontsize=8)
ax_ap.set_facecolor(BG)
ax_ap.grid(axis='y', alpha=0.3)
for rect in ax_ap.patches:
    ax_ap.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 2,
               str(int(rect.get_height())), ha='center', va='bottom', fontsize=9)

# Nota final por nivel dificultad
ax_niv = fig.add_subplot(gs[2, 1])
order = df.groupby('nivel_dificultad')['nota_final'].median().sort_values().index
sns.boxplot(data=df, x='nivel_dificultad', y='nota_final', order=order,
            palette='Set2', ax=ax_niv)
ax_niv.set_title('Nota Final por Dificultad', fontsize=10, fontweight='bold')
ax_niv.set_xlabel('Nivel', fontsize=8)
ax_niv.set_ylabel('Nota Final', fontsize=8)
ax_niv.set_facecolor(BG)
ax_niv.grid(axis='y', alpha=0.3)

plt.savefig('/home/claude/fig1_eda.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n[✓] Figura 1 (EDA) guardada.")

# ─── 3. PREPROCESAMIENTO ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. PREPROCESAMIENTO")
print("="*60)

df_prep = df.copy()

# Imputar nulos categóricos PRIMERO (antes del encoding)
for col in ['estilo_aprendizaje', 'horario_estudio_preferido']:
    moda = df_prep[col].dropna().mode()[0]
    n_null = df_prep[col].isnull().sum()
    df_prep[col] = df_prep[col].fillna(moda)
    print(f"  '{col}': {n_null} nulos imputados con moda = '{moda}'")

# Imputar nulos numéricos con mediana
for col in df_prep.select_dtypes(include='number').columns:
    n_null = df_prep[col].isnull().sum()
    if n_null > 0:
        med = df_prep[col].median()
        df_prep[col] = df_prep[col].fillna(med)
        print(f"  '{col}': {n_null} nulos imputados con mediana = {med:.2f}")

# Encoding variables categóricas (tras imputación, no habrá NaN)
cat_cols = ['nivel_dificultad', 'tiene_tutor', 'horario_estudio_preferido', 'estilo_aprendizaje']
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_prep[col + '_enc'] = le.fit_transform(df_prep[col].astype(str))
    encoders[col] = le
    print(f"  '{col}' → encoded ({dict(zip(le.classes_, le.transform(le.classes_)))})")

print(f"\nShape tras preprocesamiento: {df_prep.shape}")
print(f"Nulos restantes: {df_prep.isnull().sum().sum()}")

# Features y targets
feature_cols = ['horas_estudio_semanal', 'nota_anterior', 'tasa_asistencia',
                'horas_sueno', 'edad',
                'nivel_dificultad_enc', 'tiene_tutor_enc',
                'horario_estudio_preferido_enc', 'estilo_aprendizaje_enc']

X = df_prep[feature_cols]
y_reg = df_prep['nota_final']
y_clf = df_prep['aprobado']

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
_, _, y_train_c, y_test_c = train_test_split(X_scaled, y_clf, test_size=0.2, random_state=42)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─── 4. REGRESIÓN LINEAL ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. MODELO DE REGRESIÓN LINEAL (nota_final)")
print("="*60)

lr = LinearRegression()
lr.fit(X_train, y_train_r)
y_pred_r = lr.predict(X_test)

mae = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")

# Coeficientes
coef_df = pd.DataFrame({'Feature': feature_cols, 'Coeficiente': lr.coef_}).sort_values('Coeficiente')
print("\nCoeficientes:")
print(coef_df.to_string(index=False))

# Figura 2: Regresión
fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=BG)
fig.suptitle('Regresión Lineal – Predicción nota_final', fontsize=14, fontweight='bold')

# Real vs predicho
ax = axes[0]
ax.scatter(y_test_r, y_pred_r, alpha=0.45, color=C1, edgecolors='none', s=30)
lims = [min(y_test_r.min(), y_pred_r.min())-2, max(y_test_r.max(), y_pred_r.max())+2]
ax.plot(lims, lims, 'r--', lw=1.5, label='Perfecta')
ax.set_xlabel('Valor Real', fontsize=10)
ax.set_ylabel('Valor Predicho', fontsize=10)
ax.set_title(f'Real vs. Predicho\nR²={r2:.3f}', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(BG)
ax.grid(alpha=0.3)

# Residuos
residuos = y_test_r - y_pred_r
ax = axes[1]
ax.scatter(y_pred_r, residuos, alpha=0.45, color=C2, edgecolors='none', s=30)
ax.axhline(0, color='red', lw=1.5, ls='--')
ax.set_xlabel('Predicho', fontsize=10)
ax.set_ylabel('Residuo', fontsize=10)
ax.set_title('Residuos vs. Predicho', fontsize=11, fontweight='bold')
ax.set_facecolor(BG)
ax.grid(alpha=0.3)

# Coeficientes
ax = axes[2]
colors = [C3 if v >= 0 else C2 for v in coef_df['Coeficiente']]
bars = ax.barh(coef_df['Feature'], coef_df['Coeficiente'], color=colors, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Coeficientes del Modelo', fontsize=11, fontweight='bold')
ax.set_xlabel('Coeficiente', fontsize=10)
ax.set_facecolor(BG)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig2_regresion.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n[✓] Figura 2 (Regresión) guardada.")

# ─── 5. REGRESIÓN LOGÍSTICA ──────────────────────────────────────────────────
print("\n" + "="*60)
print("5. MODELO DE REGRESIÓN LOGÍSTICA (aprobado)")
print("="*60)

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train_c)
y_pred_c = log_reg.predict(X_test)
y_prob_c = log_reg.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test_c, y_pred_c)
print(f"  Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c, target_names=['Suspendido', 'Aprobado']))

fpr, tpr, _ = roc_curve(y_test_c, y_prob_c)
roc_auc = auc(fpr, tpr)
print(f"  AUC-ROC: {roc_auc:.4f}")

# Figura 3: Clasificación
fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor=BG)
fig.suptitle('Regresión Logística – Clasificación aprobado', fontsize=14, fontweight='bold')

# Matriz de confusión
ax = axes[0]
cm = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Suspendido', 'Aprobado'],
            yticklabels=['Suspendido', 'Aprobado'],
            linewidths=0.5, cbar=False, annot_kws={'size': 13})
ax.set_title(f'Matriz de Confusión\nAccuracy={acc:.3f}', fontsize=11, fontweight='bold')
ax.set_ylabel('Real', fontsize=10)
ax.set_xlabel('Predicho', fontsize=10)

# ROC
ax = axes[1]
ax.plot(fpr, tpr, color=C1, lw=2, label=f'AUC = {roc_auc:.3f}')
ax.plot([0, 1], [0, 1], color=GRAY, lw=1.5, ls='--', label='Random')
ax.fill_between(fpr, tpr, alpha=0.1, color=C1)
ax.set_xlabel('FPR (1 - Especificidad)', fontsize=10)
ax.set_ylabel('TPR (Sensibilidad)', fontsize=10)
ax.set_title('Curva ROC', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor(BG)
ax.grid(alpha=0.3)

# Importancia (coeficientes logísticos)
coef_log = pd.DataFrame({'Feature': feature_cols, 'Coeficiente': log_reg.coef_[0]}).sort_values('Coeficiente')
ax = axes[2]
colors_l = [C3 if v >= 0 else C2 for v in coef_log['Coeficiente']]
ax.barh(coef_log['Feature'], coef_log['Coeficiente'], color=colors_l, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Coeficientes Logísticos', fontsize=11, fontweight='bold')
ax.set_xlabel('Coeficiente', fontsize=10)
ax.set_facecolor(BG)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/fig3_clasificacion.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n[✓] Figura 3 (Clasificación) guardada.")

print("\n" + "="*60)
print("EJECUCIÓN COMPLETADA ✓")
print("="*60)
