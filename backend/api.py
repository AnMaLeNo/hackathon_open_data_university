from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from contextlib import asynccontextmanager
from typing import Dict, Any

# Import de tes propres fonctions
from qdrant_tools import rechercher_reactions_similaires
from analyse import modeliser_recompense_semantique

# --- Variables Globales ---
ml_models = {}
qdrant_db = {}

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
    
    print("✅ Serveur prêt à recevoir des requêtes !")
    yield
    
    # Nettoyage à l'arrêt du serveur
    print("🛑 Arrêt du serveur : Libération des ressources.")
    ml_models.clear()
    qdrant_db.clear()

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
    limit: int = 1000 # On met la même limite par défaut que dans ton main.py

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
            limit=request.limit
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