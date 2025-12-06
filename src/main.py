# src/main.py

import os

from data_loader import DataLoader
from recommender import ItemItemRecommender, PopularityRecommender


def main():
    # ------------------------------------------------------------------ #
    # 1. Chargement des données
    # ------------------------------------------------------------------ #
    loader = DataLoader(data_path="data")
    interactions, items, sample = loader.get_all_data()

    # Colonnes du sample Kaggle
    user_col_sub = sample.columns[0]      # normalement 'user_id'
    rec_col_sub = sample.columns[1]       # normalement 'recommendation'

    print(f"[main] Colonne user du sample : {user_col_sub}")
    print(f"[main] Colonne reco du sample : {rec_col_sub}")

    # ------------------------------------------------------------------ #
    # 2. Modèle de popularité (fallback + baseline)
    # ------------------------------------------------------------------ #
    pop_model = PopularityRecommender(k=50)
    pop_model.fit(interactions, item_col="i")
    popular_items = pop_model.recommend(k=50)
    print(f"[main] Quelques items populaires : {popular_items[:10]}")

    # ------------------------------------------------------------------ #
    # 3. Modèle item–item (avec récence, et SANS exclure les livres déjà vus)
    # ------------------------------------------------------------------ #
    item_item_model = ItemItemRecommender(
        n_recent=5,  # on prend les 5 derniers livres pour chaque user
    )
    item_item_model.fit(interactions, user_col="u", item_col="i")

    # ------------------------------------------------------------------ #
    # 4. Prédictions pour tous les utilisateurs du sample
    # ------------------------------------------------------------------ #
    users_to_predict = sample[user_col_sub].tolist()
    print(f"[main] Nombre d'utilisateurs à prédire : {len(users_to_predict)}")

    submission = item_item_model.predict_for_users(
        user_ids=users_to_predict,
        user_col_name=user_col_sub,
        items_list_col_name=rec_col_sub,
        k=10,
        popularity_fallback=popular_items,
    )

    # ------------------------------------------------------------------ #
    # 5. Sauvegarde au format Kaggle
    # ------------------------------------------------------------------ #
    os.makedirs("submissions", exist_ok=True)
    out_path = os.path.join("submissions", "submission_item_item_recency.csv")
    submission.to_csv(out_path, index=False)
    print(f"\n✅ Fichier de submission généré : {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()