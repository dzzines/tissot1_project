# src/data_loader.py

import os
import pandas as pd


class DataLoader:
    """
    Chargement et préparation des données du projet Lazy Librarian.
    """

    def __init__(self, data_path: str = "data"):
        """
        :param data_path: dossier contenant les fichiers CSV Kaggle.
        """
        self.data_path = data_path
        self.interactions: pd.DataFrame | None = None
        self.items: pd.DataFrame | None = None
        self.submission_format: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    #  Fonctions privées utilitaires
    # ------------------------------------------------------------------ #

    @staticmethod
    def _standardise_interactions_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        - Trouve les colonnes user/item et les renomme en 'u' et 'i'
        - Essaie de trouver une colonne temporelle ('timestamp', 'time', etc.)
          ou sinon crée un index 'ts'
        - Trie les interactions par utilisateur + temps
        """
        if df.empty:
            raise ValueError("Le DataFrame des interactions est vide.")

        lower_to_original = {c.lower(): c for c in df.columns}

        user_candidates = ["u", "user_id", "userid", "user", "uid"]
        item_candidates = ["i", "item_id", "itemid", "item", "book_id", "bid"]

        user_col = None
        item_col = None

        for c in user_candidates:
            if c in lower_to_original:
                user_col = lower_to_original[c]
                break

        for c in item_candidates:
            if c in lower_to_original:
                item_col = lower_to_original[c]
                break

        if user_col is None or item_col is None:
            raise ValueError(
                f"Impossible de trouver les colonnes user/item dans {df.columns.tolist()}"
            )

        # Renommage en 'u' et 'i'
        rename_map: dict[str, str] = {}
        if user_col != "u":
            rename_map[user_col] = "u"
        if item_col != "i":
            rename_map[item_col] = "i"

        df = df.rename(columns=rename_map)

        # Recherche d'une colonne temporelle
        time_candidates = ["timestamp", "ts", "time", "t", "date", "datetime"]
        time_col_original = None
        for c in time_candidates:
            if c in lower_to_original:
                time_col_original = lower_to_original[c]
                break

        if time_col_original is not None and time_col_original in df.columns:
            if time_col_original != "ts":
                df = df.rename(columns={time_col_original: "ts"})
        else:
            # Si pas de colonne temporelle → on crée un ordre implicite
            df["ts"] = range(len(df))

        # Tri par utilisateur + temps
        df = df.sort_values(by=["u", "ts"]).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------ #
    #  Chargement des fichiers
    # ------------------------------------------------------------------ #

    def load_interactions(self) -> pd.DataFrame:
        """
        Charge interactions_train.csv et renomme les colonnes en 'u' et 'i'.
        Garantit un ordre temporel par utilisateur.
        """
        path = os.path.join(self.data_path, "interactions_train.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier non trouvé : {path}")

        df = pd.read_csv(path)
        df = self._standardise_interactions_columns(df)

        self.interactions = df
        print(
            f"[DataLoader] Interactions chargées : {len(df)} lignes, "
            f"{df['u'].nunique()} users, {df['i'].nunique()} items."
        )
        return self.interactions

    def load_items(self) -> pd.DataFrame | None:
        """
        Charge items.csv (métadonnées des livres) si présent.
        """
        path = os.path.join(self.data_path, "items.csv")
        if not os.path.exists(path):
            print(f"[DataLoader] Attention : items.csv introuvable à {path}.")
            self.items = None
            return None

        df = pd.read_csv(path)
        self.items = df
        print(f"[DataLoader] Items chargés : {len(df)} livres.")
        return self.items

    def load_submission_sample(self) -> pd.DataFrame:
        """
        Charge sample_submission.csv pour récupérer le format de soumission.
        """
        path = os.path.join(self.data_path, "sample_submission.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier non trouvé : {path}")

        df = pd.read_csv(path)
        self.submission_format = df
        print(f"[DataLoader] sample_submission.csv chargé : {len(df)} lignes.")
        return self.submission_format

    # ------------------------------------------------------------------ #

    def get_all_data(self):
        """
        Charge d'un coup toutes les données principales.
        :return: (interactions, items, submission_format)
        """
        interactions = self.load_interactions()
        items = self.load_items()
        submission_format = self.load_submission_sample()
        return interactions, items, submission_format