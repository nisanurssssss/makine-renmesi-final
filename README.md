# Uçtan Uca Makine Öğrenmesi Projesi: Meme Kanseri Teşhis Tahmini

Bu çalışma; Breast Cancer (Meme Kanseri) veri seti üzerinde veri keşfi, öznitelik mühendisliği, öznitelik seçimi, çoklu model eğitimi, GridSearchCV ile hiperparametre optimizasyonu ve SHAP tabanlı model açıklanabilirliği adımlarını içeren uçtan uca bir sınıflandırma projesidir.



##  Projenin Amacı ve Kapsamı
- **Veri Analizi:** Sayısal özniteliklerin ve hedef değişken dağılımının incelenmesi.
- **Öznitelik Mühendisliği:** Hücre çekirdeği geometrisine dayalı yeni oranlar türetilmesi.
- **Öznitelik Seçimi:** Mutlak korelasyon filtresi ile en ayırt edici değişkenlerin seçilmesi.
- **Model Karşılaştırması:** Logistic Regression, Decision Tree ve Random Forest modellerinin validation başarımı üzerinden kıyaslanması.
- **Hiperparametre Optimizasyonu:** Random Forest modeli için 5 katlı çapraz doğrulama (5-Fold CV) ve GridSearchCV kullanımı.
- **Model Açıklanabilirliği (XAI):** SHAP (TreeExplainer) ile özniteliklerin model kararlarına etkisinin analizi.

---

##  Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. **Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install pandas numpy scikit-learn shap matplotlib
