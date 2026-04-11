from typing import Dict, List, Any, Union

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

def modeliser_recompense_semantique(
    voisinage_k: List[Dict[str, Any]],
    vecteur_theta: Dict[str, float] = None,
    vecteur_lambda: Dict[str, float] = None
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Calcule la fonction de récompense sémantique, génère le score prédictif S_hat_m,
    et quantifie la cardinalité du support statistique pour chaque modèle de langage.
    
    Args:
        voisinage_k: Liste de dictionnaires (sortie de rechercher_reactions_similaires).
        vecteur_theta: Dictionnaire des pondérations \theta_j pour les signaux positifs.
        vecteur_lambda: Dictionnaire des pénalités \lambda_k pour les signaux négatifs.
        
    Returns:
        Un dictionnaire associant chaque identifiant de modèle à une structure de 
        données contenant le score sémantique agrégé (float) et le volume d'évaluations (int).
    """
    
    # 1. Initialisation des tenseurs de pondération
    if vecteur_theta is None:
        vecteur_theta = {
            "liked": 1.0,
            "useful": 1.5,
            "creative": 1.2,
            "clear_formatting": 0.5 
        }
        
    if vecteur_lambda is None:
        vecteur_lambda = {
            "disliked": 1.0,
            "incorrect": 2.5, 
            "superficial": 1.0,
            "instructions_not_followed": 2.0 
        }

    # Structure d'agrégation incluant le compteur de cardinalité
    donnees_agregation_m = {}

    # 2. Itération sur l'ensemble K
    for interaction in voisinage_k:
        modele_m = interaction.get("refers_to_model")
        if not modele_m:
            continue

        similarite_wi = interaction.get("score", 0.0)
        recompense_ri = 0.0

        # Opérateur de sommation (attributs positifs)
        for signal_j, theta_j in vecteur_theta.items():
            if interaction.get(signal_j) is True:
                recompense_ri += theta_j

        # Opérateur de soustraction (attributs négatifs)
        for signal_k, lambda_k in vecteur_lambda.items():
            if interaction.get(signal_k) is True:
                recompense_ri -= lambda_k

        # 3. Accumulation matricielle et incrémentation du support
        if modele_m not in donnees_agregation_m:
            donnees_agregation_m[modele_m] = {
                "numerateur_somme_ponderee": 0.0,
                "denominateur_somme_poids": 0.0,
                "volume_support": 0
            }

        donnees_agregation_m[modele_m]["numerateur_somme_ponderee"] += similarite_wi * recompense_ri
        donnees_agregation_m[modele_m]["denominateur_somme_poids"] += similarite_wi
        donnees_agregation_m[modele_m]["volume_support"] += 1

    # 4. Inférence probabiliste finale et structuration de la sortie
    resultats_analytiques = {}
    
    for modele_m, metriques in donnees_agregation_m.items():
        if metriques["denominateur_somme_poids"] > 0:
            score_hat_s = metriques["numerateur_somme_ponderee"] / metriques["denominateur_somme_poids"]
            resultats_analytiques[modele_m] = {
                "score_semantique": round(score_hat_s, 4),
                "volume_support": metriques["volume_support"]
            }
        else:
            resultats_analytiques[modele_m] = {
                "score_semantique": 0.0,
                "volume_support": 0
            }

    return resultats_analytiques