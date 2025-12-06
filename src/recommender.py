# src/recommender.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class PopularityRecommender:
    """
    Baseline simple : recommande les livres les plus empruntés globalement.
    Sert de fallback si on n'a pas d'historique utilisateur.
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.top_items: list[int] | None = None

    def fit(self, interactions_df: pd.DataFrame, item_col: str = "i") -> None:
        """
        :param interactions_df: DataFrame contenant au moins une colonne item_col.
        :param item_col: nom de la colonne item (par défaut 'i').
        """
        counts = interactions_df[item_col].value_counts()
        self.top_items = counts.index.astype(int).tolist()
        print(f"[PopularityRecommender] Top items appris ({len(self.top_items)} livres).")

    def recommend(self, k: int | None = None) -> list[int]:
        if self.top_items is None:
            raise RuntimeError("Appeler fit() avant recommend().")
        if k is None:
            k = self.k
        return self.top_items[:k]


class ItemItemRecommender:
    """
    Filtrage collaboratif item–item :
    - construit une matrice user–item
    - calcule une similarité cosinus entre livres
    - pour chaque utilisateur, prend ses derniers livres lus et recommande les plus similaires

    ⚠️ On a le droit de recommander des livres déjà consultés par l'utilisateur.
    On NE filtre donc PAS les items déjà vus.
    """

    def __init__(self, n_recent: int = 5):
        """
        :param n_recent: nombre de livres récents utilisés pour scorer les autres.
        """
        self.n_recent = n_recent

        self.user_ids: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.user_id_to_index: dict[int, int] | None = None
        self.item_id_to_index: dict[int, int] | None = None

        self.interaction_matrix: csr_matrix | None = None  # (n_users, n_items)
        self.item_similarity: np.ndarray | None = None      # (n_items, n_items)

        # historique ordonné : user_id externe -> liste d'indices d'items (internes)
        self.user_histories: dict[int, list[int]] | None = None

    # ------------------------------------------------------------------ #

    def _build_mappings(self, interactions_df: pd.DataFrame,
                        user_col: str = "u",
                        item_col: str = "i") -> None:
        self.user_ids = interactions_df[user_col].astype(int).unique()
        self.item_ids = interactions_df[item_col].astype(int).unique()

        self.user_id_to_index = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.item_id_to_index = {iid: idx for idx, iid in enumerate(self.item_ids)}

    # ------------------------------------------------------------------ #

    def fit(self,
            interactions_df: pd.DataFrame,
            user_col: str = "u",
            item_col: str = "i") -> None:
        """
        Entraîne le modèle item–item.
        Suppose que interactions_df est déjà trié par (user, temps).
        """
        if interactions_df.empty:
            raise ValueError("interactions_df est vide dans ItemItemRecommender.fit().")

        print("[ItemItemRecommender] Construction des mappings...")
        self._build_mappings(interactions_df, user_col=user_col, item_col=item_col)

        n_users = len(self.user_ids)
        n_items = len(self.item_ids)
        print(f"[ItemItemRecommender] {n_users} users, {n_items} items.")

        # Matrice user–item
        print("[ItemItemRecommender] Construction de la matrice sparse user–item...")
        user_indices = interactions_df[user_col].astype(int).map(self.user_id_to_index).to_numpy()
        item_indices = interactions_df[item_col].astype(int).map(self.item_id_to_index).to_numpy()
        data = np.ones(len(interactions_df), dtype=np.float32)

        self.interaction_matrix = csr_matrix(
            (data, (user_indices, item_indices)),
            shape=(n_users, n_items),
        )

        # Historique utilisateur (ordonné)
        print("[ItemItemRecommender] Construction des historiques utilisateurs...")
        self.user_histories = {}
        # interactions_df est déjà trié dans DataLoader
        for uid, iid in zip(interactions_df[user_col].astype(int),
                            interactions_df[item_col].astype(int)):
            i_idx = self.item_id_to_index[iid]
            self.user_histories.setdefault(uid, []).append(i_idx)

        # Similarité item–item
        print("[ItemItemRecommender] Calcul de la similarité item–item (cosinus)...")
        # On binarise les interactions pour éviter de sur-pondérer les gros lecteurs
        binary_matrix = (self.interaction_matrix > 0).astype(np.float32)
        item_matrix = binary_matrix.T  # (n_items, n_users)

        self.item_similarity = cosine_similarity(item_matrix)
        print("[ItemItemRecommender] Entraînement terminé (matrice de similarité calculée).")

    # ------------------------------------------------------------------ #

    def recommend_for_user(self,
                           user_id: int,
                           k: int = 10,
                           popularity_fallback: list[int] | None = None) -> list[int]:
        """
        Recommande k livres pour un utilisateur donné (id externe).

        ⚠️ On NE filtre PAS les livres déjà vus :
        les interactions déjà faites peuvent être recommandées à nouveau.
        """
        if (self.item_similarity is None or
                self.user_histories is None or
                self.item_ids is None):
            raise RuntimeError("Appeler fit() avant recommend_for_user().")

        try:
            uid_int = int(user_id)
        except Exception:
            uid_int = int(str(user_id))

        history_indices = self.user_histories.get(uid_int)

        # Aucun historique -> fallback popularité
        if not history_indices:
            if popularity_fallback:
                return popularity_fallback[:k]
            return []

        history_indices = np.array(history_indices, dtype=int)

        # On prend les n derniers livres (récence)
        recent_indices = history_indices[-self.n_recent:]
        n_used = len(recent_indices)

        # Pondération par récence : dernier = poids max, plus ancien = plus faible
        # Exemple : pour 3 items -> [1.0, 0.7, 0.4]
        weights = np.linspace(1.0, 0.4, num=n_used, endpoint=True, dtype=np.float32)

        # Score = somme pondérée des similarités des items récents
        scores = np.zeros(self.item_similarity.shape[0], dtype=np.float32)
        for idx, w in zip(recent_indices, weights):
            scores += w * self.item_similarity[idx]

        # Si rien de valable -> fallback
        if not np.isfinite(scores).any():
            if popularity_fallback:
                return popularity_fallback[:k]
            return []

        top_k = min(k, scores.shape[0])
        candidate_indices = np.argpartition(scores, -top_k)[-top_k:]
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])][::-1]
        candidate_indices = candidate_indices[:k]

        rec_ids = [int(self.item_ids[i]) for i in candidate_indices]

        # Compléter avec popularité si moins que k
        if popularity_fallback is not None and len(rec_ids) < k:
            extra = [it for it in popularity_fallback if it not in rec_ids]
            rec_ids = rec_ids + extra[: (k - len(rec_ids))]

        return rec_ids

    # ------------------------------------------------------------------ #

    def predict_for_users(self,
                          user_ids,
                          user_col_name: str = "user_id",
                          items_list_col_name: str = "recommendation",
                          k: int = 10,
                          popularity_fallback: list[int] | None = None) -> pd.DataFrame:
        """
        Génère un DataFrame au format Kaggle :
        colonnes = [user_col_name, items_list_col_name]
        avec k ids séparés par des espaces par utilisateur.
        """
        rows: list[dict] = []

        for uid in user_ids:
            recs = self.recommend_for_user(
                user_id=uid,
                k=k,
                popularity_fallback=popularity_fallback,
            )
            if not recs and popularity_fallback is not None:
                recs = popularity_fallback[:k]

            recs = recs[:k]
            rows.append({
                user_col_name: int(uid),
                items_list_col_name: " ".join(str(x) for x in recs),
            })

        df = pd.DataFrame(rows)
        return df