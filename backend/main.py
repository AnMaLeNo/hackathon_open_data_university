from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json

from qdrant_tools import rechercher_reactions_similaires, indexer_corpus, indexer_corpus_generique
from analyse import modeliser_recompense_semantique, optimiser_routage_topsis

import numpy as np
from qdrant_client.models import Distance

def index_all_corpus(client, dim_vecteur):
    colonnes_payload_cibles = [
        "question_content", 
        "conversation_pair_id", 
        "refers_to_model", 
        "model_pos",
        "liked", 
        "disliked", 
        "comment", 
        "useful", 
        "creative", 
        "clear_formatting", 
        "superficial", 
        "instructions_not_followed", 
        "incorrect"
    ]
    indexer_corpus_generique(
        client=client, # Instance pré-existante du client Qdrant
        vector_file_path="./base_vectorielle/base_vectorielle_reactions_question_content.parquet",
        metadata_file_path="./database/reactions.parquet",
        collection_name="index_reactions_question_content",
        vector_size=dim_vecteur,
        vector_column="embedding", # Paramètre par défaut, rendu explicite ici
        join_key="id",             # Paramètre par défaut, rendu explicite ici
        payload_columns=colonnes_payload_cibles,
        index_fields=["conversation_pair_id"],
        batch_size=1000,           # Correspond au seuil de bufferisation initial
        distance_metric=Distance.COSINE
    )

    colonnes_payload_cibles = [
        "question_content", 
        "conversation_pair_id", 
        "refers_to_model", 
        "model_pos",
        "liked", 
        "disliked", 
        "comment", 
        "useful", 
        "creative", 
        "clear_formatting", 
        "superficial", 
        "instructions_not_followed", 
        "incorrect"
    ]
    indexer_corpus_generique(
        client=client, # Instance pré-existante du client Qdrant
        vector_file_path="./base_vectorielle/base_vectorielle_reactions_comment.parquet",
        metadata_file_path="./database/reactions.parquet",
        collection_name="index_reactions_comment",
        vector_size=dim_vecteur,
        vector_column="embedding", # Paramètre par défaut, rendu explicite ici
        join_key="id",             # Paramètre par défaut, rendu explicite ici
        payload_columns=colonnes_payload_cibles,
        index_fields=["conversation_pair_id"],
        batch_size=1000,           # Correspond au seuil de bufferisation initial
        distance_metric=Distance.COSINE
    )


def main():
    # --- INITIALISATION ---
    model = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(url="http://localhost:6333")
    dim_vecteur = model.get_embedding_dimension()

    with open('metriques_physiques.json', 'r', encoding='utf-8') as fichier:
        # On charge le contenu dans la variable 'data'
        metriques_physiques = json.load(fichier)

    #indexer_corpus(client, "index_reactions", dim_vecteur)
    index_all_corpus(client, dim_vecteur)
    print("Tape ta requête (ou 'exit' pour quitter)\n")

    while True:
        # --- INPUT UTILISATEUR ---
        requete = input(">> ").strip()

        if requete.lower() in {"exit", "quit"}:
            print("Fin du programme.")
            break

        if not requete:
            continue

        # input conversation_id
        conversation_id = input(">> ").strip()

        # --- ENCODAGE ---
        vecteur = model.encode(requete, convert_to_tensor=False).tolist()

        # --- RECHERCHE ---
        resultats = rechercher_reactions_similaires(
            client=client,
            vecteur_requete=vecteur,
            collection_name="index_reactions_question_content",
            limit=1000
        )
        print(json.dumps(resultats, indent=2, ensure_ascii=False))
        print("\n" + "-" * 50 + "\n")
        analyse = modeliser_recompense_semantique(resultats)
        print("resultat de l'analyse: " + json.dumps(analyse, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()