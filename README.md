---
title: HRPredict
sdk: docker
app_port: 7860
---

# HRPredict – Employee Attrition Prediction (MLOps)

HRPredict est une application de machine learning permettant de prédire la probabilité de départ d’un employé à partir de ses caractéristiques professionnelles et personnelles.

Ce projet met en œuvre une chaîne MLOps complète : développement du modèle, exposition via API, interface utilisateur, stockage des prédictions, conteneurisation et déploiement continu.

---

## Démo en ligne

Application déployée :
https://huggingface.co/spaces/basmoket/hrpredict

---

## Objectifs du projet

* Prédire le risque de départ (classification binaire)
* Fournir une interface simple pour la saisie des données
* Exposer le modèle via une API REST
* Enregistrer les requêtes et résultats pour traçabilité
* Mettre en place une architecture reproductible et déployable

---

## Architecture

### En local (Docker Compose)

Streamlit → FastAPI → PostgreSQL (Docker) → Modèle scikit-learn

### En production (Hugging Face)

Streamlit → FastAPI → PostgreSQL cloud (Neon) → Modèle scikit-learn

Base de données cloud :
https://neon.tech

Les prédictions et les données d’entrée sont persistées dans Neon afin d’assurer la traçabilité en environnement déployé.

---

## Technologies utilisées

### Machine Learning

[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas\&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy\&logoColor=white)](https://numpy.org)

---

### Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic\&logoColor=white)](https://docs.pydantic.dev)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?logo=sqlalchemy\&logoColor=white)](https://www.sqlalchemy.org)

---

### Frontend

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit\&logoColor=white)](https://streamlit.io)

---

### Base de données

[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql\&logoColor=white)](https://www.postgresql.org)
[![Neon](https://img.shields.io/badge/Neon-000000?logo=postgresql\&logoColor=white)](https://neon.tech)

---

### MLOps / DevOps

[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker\&logoColor=white)](https://www.docker.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github\&logoColor=white)](https://github.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?logo=github-actions\&logoColor=white)](https://docs.github.com/en/actions)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFCC00?logo=huggingface\&logoColor=black)](https://huggingface.co/spaces)


---

## Installation et exécution en local

### 1. Cloner le dépôt

```bash
git clone https://github.com/basileguerin/hrpredict.git
cd hrpredict
```

### 2. Lancer l’application

```bash
docker compose up --build
```

Services démarrés :

* PostgreSQL : localhost:5432
* API FastAPI : http://localhost:8000
* Interface Streamlit : http://localhost:7860

Documentation interactive de l’API :
http://localhost:8000/docs

---

## Utilisation de l’application

1. Ouvrir l’interface Streamlit
2. Renseigner les informations de l’employé
3. Cliquer sur **Predict**
4. L’application affiche :

   * la probabilité de départ
   * la décision finale
5. La requête et le résultat sont enregistrés en base PostgreSQL

---

## Organisation du dépôt

```
.
├── api/
│   ├── main.py          # API FastAPI
│   └── db.py            # Connexion PostgreSQL
│
├── db/
│   ├── create_db.py     # Initialisation des tables
│   └── schema.sql
│
├── model/
│   └── classifier_employee.pkl
│
├── data/
│   └── dataset_clean.csv
│
├── tests/
│   └── test_ci.py       # Tests unitaires et fonctionnels
│
├── app.py               # Interface Streamlit
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Modèle de machine learning

Problème : classification binaire (départ / non départ)
Algorithme : Régression Logistique

Prétraitement :

* normalisation des variables numériques
* endodage des variables catégorielles

Performances (jeu de test) :

* ROC AUC : ~0.85
* Précision : ~0.80
* Rappel : ~0.70

Le seuil de décision est optimisé pour équilibrer faux positifs et faux négatifs.

---

## Tests et qualité

Suite de tests avec pytest :

* Tests unitaires des endpoints
* Validation des cas d’erreur :

  * features manquantes
  * valeurs invalides (None, NaN, Inf)
* Tests fonctionnels du pipeline complet

Couverture du code API : ~94%

Intégration continue :

* exécution automatique des tests sur push (GitHub Actions)
* rapport de couverture ajouté au résumé du job

---

## Déploiement

Déploiement automatique sur Hugging Face Spaces :

1. Push sur la branche `main`
2. Build Docker
3. Lancement du conteneur
4. Connexion à la base PostgreSQL Neon

Le conteneur exécute :

* FastAPI (backend)
* Streamlit (interface)

---

## Contact

Basile
LinkedIn : https://www.linkedin.com/in/basile-guerin-66b333255/
Email : basile.guerin1@gmail.com