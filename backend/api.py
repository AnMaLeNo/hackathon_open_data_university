from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import json
import numpy as np

# Import de tes propres fonctions
from qdrant_tools import rechercher_reactions_similaires
from analyse import modeliser_recompense_semantique, optimiser_routage_topsis

# --- Variables Globales ---
ml_models = {}
qdrant_db = {}
app_data = {}

# --- Gestion de la durée de vie (Lifespan) ---
# C'est la méthode moderne de FastAPI pour charger les modèles lourds au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Démarrage du serveur : Chargement du modèle SentenceTransformer...")
    ml_models["encoder"] = SentenceTransformer("BAAI/bge-m3")
    
    import os
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"⏳ Démarrage du serveur : Connexion à Qdrant sur {qdrant_url}...")
    qdrant_db["client"] = QdrantClient(url=qdrant_url)
    
    print("⏳ Démarrage du serveur : Chargement des métriques physiques...")
    with open('metriques_physiques.json', 'r', encoding='utf-8') as fichier:
        app_data["metriques_physiques"] = json.load(fichier)
        
    print("✅ Serveur prêt à recevoir des requêtes !")
    yield
    
    # Nettoyage à l'arrêt du serveur
    print("🛑 Arrêt du serveur : Libération des ressources.")
    ml_models.clear()
    qdrant_db.clear()
    app_data.clear()

# --- Initialisation de l'API ---
app = FastAPI(
    title="API IA Culturelles - Routeur Sémantique",
    description="API permettant d'évaluer le meilleur modèle LLM pour un prompt donné.",
    lifespan=lifespan
)

# --- Schémas de données (Pydantic) ---
# Définit la structure exacte attendue en entrée (le JSON que le front-end enverra)
class PromptRequest(BaseModel):
    prompt: str
    limit: int = 1000 
    score_threshold: float = 0.65

class RoutageRequest(BaseModel):
    prompt: str
    # Matrice AHP 3x3 par défaut (Sémantique, Énergie, Souveraineté)
    matrice_ahp: List[List[float]] 
    limit: int = 1000
    score_threshold: float = 0.65

# --- Endpoints ---
@app.post("/api/evaluer_prompt")
async def evaluer_prompt(request: PromptRequest):
    """
    Reçoit un prompt, cherche les réactions similaires et calcule la récompense sémantique.
    """
    try:
        model = ml_models["encoder"]
        client = qdrant_db["client"]

        # 1. Encodage du prompt
        vecteur = model.encode(request.prompt, convert_to_tensor=False).tolist()

        # 2. Recherche Vectorielle (Qdrant)
        resultats = rechercher_reactions_similaires(
            client=client,
            vecteur_requete=vecteur,
            collection_name="index_reactions_question_content",
            limit=request.limit,
            score_threshold=request.score_threshold
        )

        # 3. Analyse Sémantique
        if not resultats:
            return {
                "message": "Aucune similarité trouvée.",
                "prompt": request.prompt,
                "recompenses": {}
            }

        analyse = modeliser_recompense_semantique(resultats)

        # 4. Retour au front-end
        return {
            "message": "Analyse réussie",
            "prompt": request.prompt,
            "recompenses": analyse
        }

    except Exception as e:
        # En cas d'erreur (Qdrant éteint, etc.), on renvoie une erreur 500 propre
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/meilleur_modele")
async def obtenir_meilleur_modele(request: RoutageRequest):
    """
    Détermine le meilleur modèle d'IA en fonction du prompt et des préférences utilisateur (AHP).
    Critères pris en compte : [Score Sémantique, kWh/token, Score Souveraineté]
    """
    try:
        model = ml_models["encoder"]
        client = qdrant_db["client"]
        metriques_physiques = app_data["metriques_physiques"]

        # 1. Encodage et Recherche Vectorielle
        vecteur = model.encode(request.prompt, convert_to_tensor=False).tolist()
        resultats = rechercher_reactions_similaires(
            client=client,
            vecteur_requete=vecteur,
            collection_name="index_reactions_question_content", # ou _comment selon ton choix
            limit=request.limit,
            score_threshold=request.score_threshold
        )

        if not resultats:
            return {
                "message": "Aucune donnée sémantique trouvée pour ce prompt.",
                "prompt": request.prompt,
                "modele_recommande": None,
                "score_topsis": None,
                "classement_complet": []
            }

        # 2. Calcul du score sémantique de base
        resultats_phase_2 = modeliser_recompense_semantique(resultats)

        # 3. Préparation pour TOPSIS
        # Conversion de la liste envoyée par le front-end en matrice Numpy
        matrice_ahp_np = np.array(request.matrice_ahp)
        
        # Définition stricte des critères utilisés dans cet ordre précis
        noms_criteres = ["score_semantique", "kwh/token", "score_souverainete"]
        
        # Directions : 1 (Maximiser sémantique), -1 (Minimiser énergie), 1 (Maximiser souveraineté)
        vecteur_directions = [1, -1, 1]

        # 4. Exécution du routage TOPSIS
        classement_final = optimiser_routage_topsis(
            resultats_phase_2=resultats_phase_2,
            metriques_physiques=metriques_physiques,
            matrice_ahp=matrice_ahp_np,
            vecteur_directions=vecteur_directions,
            noms_criteres=noms_criteres
        )

        if not classement_final:
            raise HTTPException(status_code=500, detail="Erreur lors du calcul TOPSIS.")

        # Le grand gagnant est le premier de la liste
        gagnant = classement_final[0]

        return {
            "prompt": request.prompt,
            "modele_recommande": gagnant[0],
            "score_topsis": gagnant[1],
            "classement_complet": classement_final
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))