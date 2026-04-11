from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import json

from tools import rechercher_reactions_similaires, indexer_corpus
from analyse import analyser_meilleure_ia, modeliser_recompense_semantique, optimiser_routage_topsis


def main():
    # --- INITIALISATION ---
    model = SentenceTransformer("BAAI/bge-m3")
    client = QdrantClient(url="http://localhost:6333")
    dim_vecteur = model.get_embedding_dimension()

    #indexer_corpus(client, "index_reactions", dim_vecteur)
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
            limit=100
        )
        print(json.dumps(resultats, indent=2, ensure_ascii=False))
        print("\n" + "-" * 50 + "\n")
        analyse = resultats_phase_2 = modeliser_recompense_semantique(resultats)
        #analyse = optimiser_routage_topsis(resultats_phase_2, {}, [], {}, {})
        print("resultat de l'analyse: " + json.dumps(analyse, indent=2, ensure_ascii=False))

        # --- RECHERCHE avec id--
        #print("\n--- Recherche avec filtre conversation_pair_id ---")
        #resultats_filtres = rechercher_reactions_similaires(
        #    client=client,
        #    vecteur_requete=vecteur,
        #    conversation_pair_id=conversation_id,
        #    limit=3
        #)
        #print(json.dumps(resultats_filtres, indent=2, ensure_ascii=False))
        #print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()