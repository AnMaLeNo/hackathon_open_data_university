def analyser_meilleure_ia(reactions_similaires: list[dict]) -> dict:
    """
    Analyse une liste de réactions issues d'une recherche de similarité sémantique
    pour calculer et recommander le meilleur modèle d'IA pour un prompt donné.
    """
    if not reactions_similaires:
        return {"erreur": "Aucune donnée de réaction similaire pour l'analyse.", "classement": []}

    # 1. Définition de la fonction de récompense (Reward Engineering)
    # Poids accordés aux signaux positifs
    poids_positifs = {
        "useful": 1.5,
        "liked": 1.0,
        "creative": 0.75,
        "clear_formatting": 0.25
    }
    
    # Pénalités accordées aux signaux négatifs
    poids_negatifs = {
        "disliked": 1.0,
        "superficial": 1.0,
        "instructions_not_followed": 1.5,
        "incorrect": 2.0
    }

    statistiques_modeles = {}

    # 2. Traitement de chaque réaction du voisinage sémantique
    for reaction in reactions_similaires:
        modele = reaction.get("refers_to_model")
        similarite_cosinus = reaction.get("score", 0.0)

        # On ignore les entrées invalides
        if not modele or similarite_cosinus == 0.0:
            continue

        # Calcul du score d'utilité (récompense) pour cette interaction précise
        score_interaction = 0.0
        
        for critere, poids in poids_positifs.items():
            if reaction.get(critere) is True:
                score_interaction += poids

        for critere, penalite in poids_negatifs.items():
            if reaction.get(critere) is True:
                score_interaction -= penalite

        # Initialisation du modèle dans le dictionnaire s'il n'existe pas
        if modele not in statistiques_modeles:
            statistiques_modeles[modele] = {
                "somme_recompenses_ponderees": 0.0,
                "somme_similarites": 0.0,
                "nombre_evaluations": 0
            }

        # 3. Application de l'agrégation pondérée (Weighted k-NN)
        # On multiplie le score de l'interaction par sa pertinence sémantique
        statistiques_modeles[modele]["somme_recompenses_ponderees"] += (score_interaction * similarite_cosinus)
        statistiques_modeles[modele]["somme_similarites"] += similarite_cosinus
        statistiques_modeles[modele]["nombre_evaluations"] += 1

    # 4. Calcul final et normalisation pour chaque modèle
    classement_final = []
    
    for modele, stats in statistiques_modeles.items():
        if stats["somme_similarites"] > 0:
            # La normalisation évite de favoriser un modèle juste parce qu'il a été testé plus souvent
            score_predictif = stats["somme_recompenses_ponderees"] / stats["somme_similarites"]
        else:
            score_predictif = 0.0

        classement_final.append({
            "modele": modele,
            "score_recommandation": round(score_predictif, 4),
            "evaluations_pertinentes_trouvees": stats["nombre_evaluations"]
        })

    # Tri du classement par score prédictif décroissant
    classement_final.sort(key=lambda x: x["score_recommandation"], reverse=True)

    # Création de l'objet de retour
    meilleure_ia = classement_final[0]["modele"] if classement_final else None
    
    return {
        "prompt_analyse": True,
        "meilleure_ia_recommandee": meilleure_ia,
        "classement_detaille": classement_final
    }