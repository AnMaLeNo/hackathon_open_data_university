import pandas as pd
import glob
import os

print("🔍 Recherche des fichiers de sauvegarde...")
# Cherche tous les fichiers qui commencent par "embeddings_sauvegarde_"
fichiers_partiels = glob.glob("embeddings_sauvegarde_*.parquet")

if not fichiers_partiels:
    print("❌ Aucun fichier trouvé. Vérifiez que vous êtes dans le bon dossier.")
else:
    print(f"✅ {len(fichiers_partiels)} fichiers trouvés ! Début de la fusion...")

    # On lit tous les petits fichiers et on les stocke dans une liste
    liste_dataframes = [pd.read_parquet(f) for f in fichiers_partiels]

    # On les colle tous ensemble de haut en bas
    df_final = pd.concat(liste_dataframes, ignore_index=True)

    # On sauvegarde le gros fichier final
    nom_fichier_final = "base_vectorielle_complete.parquet"
    df_final.to_parquet(nom_fichier_final)

    print(f"🎉 Fusion terminée ! Fichier créé : {nom_fichier_final}")
    print(f"📊 Nombre total de lignes : {len(df_final)}")