from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range, PointStruct, PayloadSchemaType, VectorParams, Distance
import json
import pandas as pd
from qdrant_client.http import models
import numpy as np



def indexer_corpus(client, collection_name="index_reactions", vector_size=768):
    """
    Fonction d'indexation avec provisionnement conditionnel de l'espace vectoriel.
    Paramètre vector_size : Doit correspondre exactement à la dimensionnalité du modèle d'embedding utilisé.
    """
    # 1. Vérification formelle et instanciation conditionnelle
    if not client.collection_exists(collection_name):
        print(f"⚙️ Création de la collection '{collection_name}' (Dimension: {vector_size}, Distance: Cosinus)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            )
        )
    else:
        collection_info = client.get_collection(collection_name)
        if collection_info.points_count > 0:
            print(f"✅ La collection contient déjà {collection_info.points_count} vecteurs. Indexation ignorée.")
            return

    print("⏳ Début de l'extraction et du chargement (ETL)...")
    
    # Chargement des structures de données en mémoire
    df_vecteurs = pd.read_parquet("./base_vectorielle/base_vectorielle_reactions_question_content.parquet", engine="pyarrow")
    df_reactions = pd.read_parquet("./database/reactions.parquet", engine="pyarrow")
    
    # Casting strict des types pour la jointure matricielle
    df_vecteurs['id'] = df_vecteurs['id'].astype(int)
    df_reactions['id'] = df_reactions['id'].astype(int)
    
    # Intersection matricielle sur la clé primaire 'id'
    df_merged = pd.merge(df_vecteurs, df_reactions, on="id", how="inner")
    
    points = []
    
    print("⏳ Création des vecteurs et ingestion par lots...")
    # Parcours itératif optimisé des tuples
    for row in df_merged.itertuples(index=False):
        
        # Contrôle d'intégrité : Rejet silencieux si l'embedding est manquant (vecteur obligatoire)
        if pd.isna(row.embedding).all() if isinstance(row.embedding, np.ndarray) else row.embedding is None:
            continue
            
        # Résolution sécurisée du tableau NumPy ou de la liste Python native
        vector_data = row.embedding.tolist() if hasattr(row.embedding, "tolist") else list(row.embedding)

        points.append(
            PointStruct(
                id=int(row.id),
                vector=vector_data,
                payload={
                    "question_content": str(row.question_content) if pd.notna(row.question_content) else "",
                    "conversation_pair_id": str(row.conversation_pair_id) if pd.notna(row.conversation_pair_id) else "",
                    "refers_to_model": str(row.refers_to_model) if pd.notna(row.refers_to_model) else "",
                    "model_pos": str(row.model_pos) if pd.notna(row.model_pos) else "",
                    
                    # Casting booléen strict avec pd.notna() pour éviter bool(np.nan) == True
                    "liked": bool(row.liked) if pd.notna(row.liked) else None,
                    "disliked": bool(row.disliked) if pd.notna(row.disliked) else None,
                    "comment": str(row.comment) if pd.notna(row.comment) else "",
                    
                    "useful": bool(row.useful) if pd.notna(row.useful) else None,
                    "creative": bool(row.creative) if pd.notna(row.creative) else None,
                    "clear_formatting": bool(row.clear_formatting) if pd.notna(row.clear_formatting) else None,
                    "superficial": bool(row.superficial) if pd.notna(row.superficial) else None,
                    "instructions_not_followed": bool(row.instructions_not_followed) if pd.notna(row.instructions_not_followed) else None,
                    "incorrect": bool(row.incorrect) if pd.notna(row.incorrect) else None
                }
            )
        )
        
        # Bufferisation et transmission réseau asynchrone (Batching)
        if len(points) >= 1000:
            client.upsert(collection_name=collection_name, points=points)
            points = []
            
    # Flush du buffer résiduel
    if points:
        client.upsert(collection_name=collection_name, points=points)
        
    print("⏳ Création de l'index de métadonnées pour 'conversation_pair_id'...")
    # Implémentation d'un index HNSW ou Keyword pour l'optimisation des requêtes conditionnelles (filtrage)
    client.create_payload_index(
        collection_name=collection_name,
        field_name="conversation_pair_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
        
    print("✅ Indexation terminée avec succès !")



# il faut implementer le fait de pouvoir fair le filtrage des des paramter exclusive !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def rechercher_reactions_similaires(
    client: QdrantClient,
    vecteur_requete: list[float],
    collection_name: str = "index_reactions",
    strict_constraints: dict = None,
    limit: int = 100,
    score_threshold: float = 0.65
) -> list[dict]:
    """
    Exécute une recherche sémantique vectorielle approximative (HNSW) couplée à 
    un filtrage strict de la charge utile (Payload Filtering).
    
    Args:
        client: Instance active du QdrantClient.
        vecteur_requete: Vecteur d'enchâssement de la requête projeté dans R^d.
        collection_name: Identifiant de la collection cible.
        strict_constraints: Dictionnaire des contraintes d'exclusion logiques.
                            Exemple: {"is_open_source": True, "energy_cost_kwh": {"lt": 150}}
        limit: Cardinalité k de l'ensemble de voisinage K retourné.
        score_threshold: Valeur minimale de similarité cosinus (w_i).
        
    Returns:
        Liste de dictionnaires documentant les métadonnées du voisinage local.
    """
    
    # 1. Construction du graphe de conditions logiques
    must_conditions = []
    
    if strict_constraints:
        for key, condition in strict_constraints.items():
            if isinstance(condition, dict):
                # Implémentation des opérateurs d'inéquation scalaire (Range)
                range_params = {}
                if "lt" in condition: range_params["lt"] = condition["lt"]
                if "gt" in condition: range_params["gt"] = condition["gt"]
                if "lte" in condition: range_params["lte"] = condition["lte"]
                if "gte" in condition: range_params["gte"] = condition["gte"]
                
                must_conditions.append(
                    FieldCondition(key=key, range=Range(**range_params))
                )
            else:
                # Implémentation de l'égalité stricte pour les booléens ou chaînes
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=condition))
                )

    # Instanciation du filtre avec intersection logique (AND)
    query_filter = Filter(must=must_conditions) if must_conditions else None

    # 2. Exécution de la projection vectorielle contrainte
    resultats_bruts = client.query_points(
        collection_name=collection_name,
        query=vecteur_requete,
        query_filter=query_filter,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True
    ).points

    # 3. Extraction et conversion des données de réaction pour l'ingénierie de récompense
    resultats_formates = []
    
    for res in resultats_bruts:
        item = {
            "id": res.id,
            "score": round(res.score, 4),
            "question_content": res.payload.get("question_content"),
            "conversation_pair_id": res.payload.get("conversation_pair_id"),
            "refers_to_model": res.payload.get("refers_to_model"),
            "model_pos": res.payload.get("model_pos"),
            
            # Variables de la fonction de récompense sémantique R(q_i, m)
            "liked": res.payload.get("liked"),
            "disliked": res.payload.get("disliked"),
            "useful": res.payload.get("useful"),
            "creative": res.payload.get("creative"),
            "clear_formatting": res.payload.get("clear_formatting"),
            "superficial": res.payload.get("superficial"),
            "instructions_not_followed": res.payload.get("instructions_not_followed"),
            "incorrect": res.payload.get("incorrect")
        }
        resultats_formates.append(item)

    return resultats_formates