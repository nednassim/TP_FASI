# 🏥 Système Intelligent de Prédiction des Complications Cardiaques

## 🌟 Fonctionnalités Principales
- **🤖 Chatbot Médical** : Interface conversationnelle pour expliquer les prédictions
- **📊 Deux Modèles Intégrés** :
  - `RandomForestRegressor` pour la prédiction quantitative
  - `DecisionTreeClassifier` pour l'analyse binaire
- **🔄 Pipeline Complet** :
  - Nettoyage automatique des données
  - Feature engineering spécialisé
  - Visualisation des résultats

## 🛠️ Installation Rapide
```bash
git clone https://github.com/nednassim/TP-FASI
cd TP-FASI
python -m venv venv # pour creer un environnment virtuel
source venv/bin/activate # pour activer le environnment virtuel
pip install -r requirements.txt # pour installer les librairies necessaires

streamlit run heart_chatbot.py # pour lancer le chatbot