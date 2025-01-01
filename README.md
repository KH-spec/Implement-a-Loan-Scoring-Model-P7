# 🏦 Implement a Loan Scoring Model
*(French version below)*

This project aims to develop a loan scoring model for the financial company **Prêt à dépenser**, which offers consumer loans to individuals with little or no credit history. The goal is to predict the probability of a client defaulting on a loan and provide an **interactive dashboard** to assist customer relationship managers.

---

## 🌟 Project Content

### 🔎 Context
- The company aims to classify loan applications as either **approved** or **rejected**.
- **Main Mission**:
  - Build a scoring model to predict the probability of client default automatically.
  - Develop an **interactive dashboard** to interpret predictions and enhance customer insights.

---

## 📊 Exploratory Analysis

### Data Distribution
- **NaN**: A significant amount of missing values was observed. Columns with more than 50% NaN were removed, and the remaining missing values were imputed using the median.
- **Target Classes**: An imbalance was observed between loans that were repaid and those that were not. This was addressed using **SMOTE oversampling**.

### Correlation Analysis
- The `EXT_SOURCE` variables have a negative correlation with the probability of default.
- The `DAYS_BIRTH` variable (age) is positively correlated with `EXT_SOURCE_1`.

---

## 🛠️ Data Preparation

1. **Data Merging**: Integrated several data sources (e.g., `application_train`, `bureau`, `credit_card_balance`).
2. **Cleaning**:
   - Replaced infinite values.
   - Encoding and standardization.
   - Removed outliers and created new features.
3. **Splitting**:
   - **Train/Test Split**: 75% training, 25% testing.
   - Balanced the dataset using SMOTE.

---

## 🤖 Modeling

### Models Trained
1. **DummyClassifier** (baseline).
2. **LogisticRegression**.
3. **RandomForestClassifier**.
4. **LGBMClassifier** (final model).

### Results
- **Best Performance**: `LGBMClassifier`, optimized using **Grid Search CV**.
- Metrics:
  - **F2-Measure**: Gave more weight to false negatives.
  - **Youden's J statistic**: Optimization based on business hypotheses.

### Interpretation
- Local analysis with **LIME**.
- Variable importance using **SHAP** to explain feature impacts.

---

## 🖥️ Interactive Dashboard

### Features
- Visualize scores and explanations for each client.
- Compare a client's information to similar groups.
- **Backend**: Flask API deployed on Heroku.
- **Frontend**: Dashboard developed using Streamlit.

### Links
- **Streamlit App**: [Dashboard Link](https://bit.ly/dashboard3hkDxiN)
- **GitHub Repositories**:
  - [Streamlit Dashboard](https://github.com/KH-spec/Implement-a-Loan-Scoring-Model-P7-Part-1)
  - [Flask API](https://github.com/KH-spec/Implement-a-Loan-Scoring-Model-P7-Part-2)

---

## 🏁 Conclusion

- **Areas for Improvement**:
  - Enhance feature engineering.
  - Explore additional techniques for class balancing.
  - Refine evaluation metrics based on business needs.
- **Deliverables**:
  - Operational loan scoring model.
  - Functional dashboard for customer relationship managers.

---

## 📚 Technologies Used

- **Python**: 3.8.8
- **Pandas**: 1.2.4
- **Seaborn**: 0.11.1
- **Matplotlib**: 3.3.4
- **LightGBM**: Latest compatible version.
- **Flask**: Backend API.
- **Streamlit**: Dashboard development.

---


### 🏦 Implémentez un Modèle de Scoring de Prêts

Ce projet vise à développer un modèle de scoring pour une société financière, **Prêt à dépenser**, qui propose des prêts à la consommation pour des personnes ayant peu ou pas d’historique de crédit. L'objectif est de prédire la probabilité qu’un client rembourse son prêt et de fournir un **Dashboard interactif** pour aider les gestionnaires de la relation client.

---

## 🌟 Contenu du Projet

### 🔎 Contexte
- La société souhaite classifier les demandes de prêts en **prêt accordé** ou **prêt refusé**.
- **Mission principale :**
  - Construire un modèle de scoring capable de prédire la probabilité de défaut d’un client.
  - Développer un **Dashboard interactif** pour interpréter les prédictions et améliorer la connaissance client.

---

## 📊 Analyse Exploratoire

### Distribution des Données
- **NaN** : Un taux important de valeurs manquantes a été observé. Les colonnes avec plus de 50 % de NaN ont été supprimées, et les autres valeurs ont été imputées par leur médiane.
- **Classes cibles** : Déséquilibre observé entre les prêts remboursés et non remboursés, traité via **SMOTE oversampling**.

### Analyse des Corrélations
- Les variables `EXT_SOURCE` ont une corrélation négative avec la probabilité de défaut.
- La variable `DAYS_BIRTH` (âge) est positivement corrélée avec `EXT_SOURCE_1`.

---

## 🛠️ Préparation des Données

1. **Fusion des tables** : Plusieurs sources de données intégrées (e.g., `application_train`, `bureau`, `credit_card_balance`).
2. **Nettoyage** :
   - Valeurs infinies remplacées.
   - Encodage et standardisation.
   - Suppression des valeurs aberrantes et création de nouvelles variables.
3. **Répartition** :
   - **Train/Test Split** : 75 % entraînement, 25 % test.
   - Équilibrage via SMOTE.

---

## 🤖 Modélisation

### Modèles Entraînés
1. **DummyClassifier** (baseline).
2. **LogisticRegression**.
3. **RandomForestClassifier**.
4. **LGBMClassifier** (modèle final).

### Résultats
- **Meilleure performance** : `LGBMClassifier`, optimisé avec **Grid Search CV**.
- Métrologie :
  - **F2-Measure** : Plus de poids donné aux faux négatifs.
  - **Youden's J statistic** : Optimisation basée sur des hypothèses métier.

### Interprétation
- Analyse locale avec **LIME**.
- Importance des variables via **SHAP** pour interpréter l’impact des caractéristiques.

---

## 🖥️ Dashboard Interactif

### Fonctionnalités
- Visualisation des scores et interprétation pour chaque client.
- Comparaison des informations d’un client avec des groupes similaires.
- **Backend** : API Flask déployée sur Heroku.
- **Frontend** : Dashboard développé avec Streamlit.

### Liens
- **App Streamlit** : [Lien vers le Dashboard](https://bit.ly/dashboard3hkDxiN)
- **Dépôts GitHub** :
  - [Streamlit Dashboard](https://github.com/KH-spec/Implement-a-Loan-Scoring-Model-P7-Part-1)
  - [Flask API](https://github.com/KH-spec/Implement-a-Loan-Scoring-Model-P7-Part-2)

---

## 🏁 Conclusion

- **Améliorations possibles** :
  - Perfectionner le Feature Engineering.
  - Tester davantage de techniques pour équilibrer les classes.
  - Approfondir l’évaluation basée sur des métriques métier.
- **Livrables** :
  - Modèle opérationnel de scoring de prêts.
  - Dashboard fonctionnel pour les gestionnaires.

---

## 📚 Technologies Utilisées

- **Python** : 3.8.8
- **Pandas** : 1.2.4
- **Seaborn** : 0.11.1
- **Matplotlib** : 3.3.4
- **LightGBM** : Dernière version compatible.
- **Flask** : Backend API.
- **Streamlit** : Développement du Dashboard.
