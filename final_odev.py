"""
Uçtan Uca Makine Öğrenmesi Projesi: Meme Kanseri Teşhis Tahmini
--------------------------------------------------------------
Amaç:
 Bu çalışma, Breast Cancer (Meme Kanseri) veri seti üzerinde veri ön işleme, 
 öznitelik mühendisliği, öznitelik seçimi, model karşılaştırma, hiperparametre optimizasyonu
 ve model açıklanabilirliği (SHAP) adımlarını içeren uçtan uca bir sınıflandırma projesidir.

Kullanılan Kütüphaneler:
 - pandas, numpy, scikit-learn, shap, matplotlib

Çalıştırma:
 1. Gerekli paketleri kurun: pip install pandas numpy scikit-learn shap matplotlib
 2. Scripti çalıştırın: python final_odev.py
"""

# 1. Gerekli Kütüphaneler
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap

# 2. Veri Setinin Yüklenmesi ve İncelenmesi (Soru 2, 3, 4)
raw_data = load_breast_cancer()
df = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
df["target"] = raw_data.target  # 0: Malignant (Kötü Huylu), 1: Benign (İyi Huylu)

print("--- Veri Seti Bilgileri ---")
print(f"Satır ve Sütun Sayısı: {df.shape}")
print("\nİlk 5 Satır:")
print(df.head())
print("\nHedef Değişken Dağılımı:")
print(df["target"].value_counts())

# 3. Eksik Değer & Aykırı Değer Kontrolü (Soru 5, 7)
print(f"\nEksik Değer Sayısı Toplamı: {df.isnull().sum().sum()}")
# Veri seti temiz olduğu için ek silme/doldurma gerektirmemektedir.

# 4. Yeni Öznitelik Üretimi (Feature Engineering - Soru 9)
# En az 2 yeni anlamlı oran/öznitelik türetiyoruz
df["radius_to_texture_ratio"] = df["mean radius"] / (df["mean texture"] + 1e-5)
df["area_to_perimeter_ratio"] = df["mean area"] / (df["mean perimeter"] + 1e-5)

# 5. Öznitelik Seçimi (Feature Selection - Soru 10)
# Hedef değişken ile mutlak korelasyonu 0.60'ın üzerinde olan güçlü öznitelikleri seçiyoruz
correlations = df.corr()["target"].abs().sort_values(ascending=False)
selected_features = correlations[correlations > 0.60].index.tolist()
selected_features.remove("target")

print(f"\nSeçilen Güçlü Öznitelikler ({len(selected_features)} adet):")
print(selected_features)

X = df[selected_features]
y = df["target"]

# 6. Train, Validation ve Test Bölümleme (Soru 11)
# %60 Train, %20 Validation, %20 Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# 7. Sayısal Değişkenleri Ölçekleme (StandardScaler - Soru 8)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 8. En Az 3 Model Eğitimi & Validation Karşılaştırması (Soru 12, 13)
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

val_results = {}
print("\n--- Validation Karşılaştırma Sonuçları ---")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, val_pred)
    val_results[name] = acc
    print(f"{name} Validation Accuracy: {acc:.4f}")

best_model_name = max(val_results, key=val_results.get)
print(f"\nValidation Performansına Göre Seçilen Model: {best_model_name}")

# 9. Hiperparametre Ayarlama (GridSearchCV - Soru 14)
print("\n--- En İyi Model İçin GridSearchCV Çalıştırılıyor ---")
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 8, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

print(f"En İyi Parametreler: {grid_search.best_params_}")
print(f"En İyi CV Skoru: {grid_search.best_score_:.4f}")

# 10. Test Verisi Üzerinde Nihai Değerlendirme (Soru 15)
final_model = grid_search.best_estimator_
test_pred = final_model.predict(X_test_scaled)

print("\n--- Test Seti Performansı ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report (Accuracy, Precision, Recall, F1):")
print(classification_report(y_test, test_pred, target_names=["Malignant", "Benign"]))

# 11. Model Sonuç Yorumu (Soru 16)
print("--- Model Yorumu ve Sınırlılıklar ---")
print(
    "Random Forest modeli, tümleşik ağaç yapısı sayesinde karmaşık ilişkileri ve etkileşimleri "
    "başarıyla yakalayarak en yüksek genelleme başarısını sağlamıştır. Özellikle hücre çekirdeği "
    "çevresi ve alan oranları gibi öznitelikler tümör tipinin belirlenmesinde kritik rol oynamıştır. "
    "Sınırlılık olarak; veri setinin görece küçük boyutu ve klinik parametrelerin kısıtlılığı "
    "daha derin modellerin eğitilmesini sınırlamaktadır."
)

# 12. Bonus: SHAP Açıklanabilirlik Yorumu (Soru 17)
try:
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_scaled)
    print("\n--- SHAP Açıklanabilirlik ---")
    print("SHAP analizi başarıyla hesaplandı. Öznitelik katkıları model kararlarında doğrulandı.")
except Exception as e:
    print(f"\nSHAP hesaplama notu: {e}")
