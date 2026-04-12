import pandas as pd
import json
import math

def get_souverainete_score(model_name: str) -> float:
    """
    Analyse le nom du modèle et retourne un score de souveraineté.
    1.0 pour les modèles français/européens, 0.0 pour les autres.
    """
    if not model_name:
        return 0.0
        
    name_lower = str(model_name).lower()
    
    # Mots-clés basés sur les modèles identifiés dans ton dataset
    mots_cles_souverains = [
        "mistral",     # Famille Mistral
        "mixtral",     # Mistral MoE
        "ministral",   # Petits modèles Mistral
        "chocolatine", # Modèle français
        "eurollm",     # Initiative européenne
        "apertus",     # Initiative open-source française
        "magistral"    # Semble être une déclinaison/finetune francophone
    ]
    
    # Si un des mots-clés est dans le nom du modèle, c'est un modèle souverain
    for mot in mots_cles_souverains:
        if mot in name_lower:
            return 1.0
            
    # Sinon, on considère que c'est un modèle étranger (US, Chinois, etc.)
    return 0.0


def extract_model_stats_to_json(df):
    """
    Extrait les statistiques des modèles d'un DataFrame de conversations
    et retourne un JSON structuré avec les paramètres, le ratio kWh/token,
    et le score de souveraineté calculé automatiquement.
    """
    # 1. Isoler les données du modèle A et les renommer avec des noms génériques
    df_a = df[[
        'model_a_name', 'model_a_total_params', 'model_a_active_params', 
        'total_conv_a_kwh', 'total_conv_a_output_tokens'
    ]].copy()
    df_a.columns = ['model_name', 'total_params', 'active_params', 'kwh', 'tokens']

    # 2. Isoler les données du modèle B et les renommer de la même façon
    df_b = df[[
        'model_b_name', 'model_b_total_params', 'model_b_active_params', 
        'total_conv_b_kwh', 'total_conv_b_output_tokens'
    ]].copy()
    df_b.columns = ['model_name', 'total_params', 'active_params', 'kwh', 'tokens']

    # 3. Fusionner les deux en un seul DataFrame vertical
    df_combined = pd.concat([df_a, df_b], ignore_index=True)
    
    # Normalisation et nettoyage des noms de modèles
    df_combined['model_name'] = df_combined['model_name'].astype(str).str.lower()
    df_combined = df_combined.dropna(subset=['model_name'])
    df_combined = df_combined[df_combined['model_name'] != 'none']

    result_dict = {}

    # 4. Grouper par nom de modèle pour récupérer les infos uniques
    for model_name, group in df_combined.groupby('model_name'):
        
        # Récupérer les paramètres (on prend la première valeur non-nulle trouvée)
        total_params = group['total_params'].dropna().iloc[0] if not group['total_params'].dropna().empty else None
        active_params = group['active_params'].dropna().iloc[0] if not group['active_params'].dropna().empty else None
        
        valid_kwh_group = group[(group['tokens'] > 0) & (group['kwh'].notna())]
        
        if not valid_kwh_group.empty:
            total_kwh = valid_kwh_group['kwh'].sum()
            total_tokens = valid_kwh_group['tokens'].sum()
            kwh_per_token = (total_kwh / total_tokens) * 1000000
        else:
            kwh_per_token = None

        # 5. Nettoyage pour le JSON et AJOUT DU SCORE DE SOUVERAINETÉ
        result_dict[str(model_name)] = {
            "total_params": float(total_params) if total_params and not math.isnan(total_params) else None,
            "active_params": float(active_params) if active_params and not math.isnan(active_params) else None,
            "kwh/token": float(kwh_per_token) if kwh_per_token and not math.isnan(kwh_per_token) else None,
            "score_souverainete": get_souverainete_score(str(model_name))
        }

    # 6. Convertir le dictionnaire en chaîne JSON formatée
    return json.dumps(result_dict, indent=4)

if __name__ == "__main__":
    # 1. Charger le fichier parquet
    df = pd.read_parquet('./database/conversations.parquet')

    # 2. Appliquer la fonction
    json_result = extract_model_stats_to_json(df)

    # 3. Afficher le résultat (ou le sauvegarder)
    print(json_result)
    
    # Décommenter les lignes ci-dessous pour sauvegarder directement le fichier
    # with open("metriques_physiques.json", "w", encoding="utf-8") as f:
    #     f.write(json_result)