from typing import Dict, List, Any

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
) -> Dict[str, float]:
    """
    Calcule la fonction de récompense sémantique et génère le score prédictif S_hat_m
    pour chaque modèle de langage présent dans le voisinage vectoriel.
    
    Args:
        voisinage_k: Liste de dictionnaires (sortie de rechercher_reactions_similaires).
        vecteur_theta: Dictionnaire des pondérations \theta_j pour les signaux positifs.
        vecteur_lambda: Dictionnaire des pénalités \lambda_k pour les signaux négatifs.
        
    Returns:
        Un dictionnaire associant chaque identifiant de modèle (str) à son 
        score sémantique agrégé \hat{S}_m (float).
    """
    
    # 1. Initialisation des tenseurs de pondération par défaut si non fournis
    # Les valeurs assignées reflètent une pondération asymétrique des signaux selon 
    # leur impact sur le succès de la tâche (Task Success) ou l'échec critique.
    if vecteur_theta is None:
        vecteur_theta = {
            "liked": 1.0,
            "useful": 1.5,
            "creative": 1.2,
            "clear_formatting": 0.5 # Pondération faible pour minorer le biais de formatage
        }
        
    if vecteur_lambda is None:
        vecteur_lambda = {
            "disliked": 1.0,
            "incorrect": 2.5, # Pénalité maximale pour hallucination factuelle
            "superficial": 1.0,
            "instructions_not_followed": 2.0 # Échec critique d'alignement formel
        }

    # Structure de données pour stocker le numérateur et le dénominateur
    # de l'équation de S_hat_m pour chaque candidat m
    donnees_agregation_m = {}

    # 2. Itération sur l'ensemble K pour le calcul de R(q_i, m)
    for interaction in voisinage_k:
        modele_m = interaction.get("refers_to_model")
        if not modele_m:
            continue

        similarite_wi = interaction.get("score", 0.0)
        recompense_ri = 0.0

        # Application de l'opérateur de sommation pour les attributs positifs
        for signal_j, theta_j in vecteur_theta.items():
            if interaction.get(signal_j) is True:
                recompense_ri += theta_j

        # Application de l'opérateur de soustraction pour les attributs négatifs
        for signal_k, lambda_k in vecteur_lambda.items():
            if interaction.get(signal_k) is True:
                recompense_ri -= lambda_k

        # 3. Accumulation pour la moyenne pondérée
        if modele_m not in donnees_agregation_m:
            donnees_agregation_m[modele_m] = {
                "numerateur_somme_ponderee": 0.0,
                "denominateur_somme_poids": 0.0
            }

        donnees_agregation_m[modele_m]["numerateur_somme_ponderee"] += similarite_wi * recompense_ri
        donnees_agregation_m[modele_m]["denominateur_somme_poids"] += similarite_wi

    # 4. Inférence probabiliste finale : Calcul de S_hat_m(q_new)
    scores_predictifs_modeles = {}
    
    for modele_m, metriques in donnees_agregation_m.items():
        if metriques["denominateur_somme_poids"] > 0:
            score_hat_s = metriques["numerateur_somme_ponderee"] / metriques["denominateur_somme_poids"]
            # Arrondi à 4 décimales pour stabiliser la précision en virgule flottante
            scores_predictifs_modeles[modele_m] = round(score_hat_s, 4)
        else:
            scores_predictifs_modeles[modele_m] = 0.0

    return scores_predictifs_modeles