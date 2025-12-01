import pandas as pd
import os

class DataLoader:
    """
    Classe responsable du chargement et de la préparation initiale des données.
    """
    
    def __init__(self, data_path='data'):
        """
        Initialise le chargeur de données.
        :param data_path: Le chemin vers le dossier contenant les fichiers CSV.
        """
        self.data_path = data_path
        self.interactions = None
        self.items = None
        self.submission_format = None

    def load_interactions(self):
        """Charge l'historique des interactions (user_id, item_id, time)."""
        print("Chargement des interactions...")
        path = os.path.join(self.data_path, 'interactions_train.csv')
        # On spécifie les types pour économiser de la mémoire et être précis
        self.interactions = pd.read_csv(path)
        print(f"Interactions chargées : {self.interactions.shape[0]} lignes.")
        return self.interactions

    def load_items(self):
        """Charge les métadonnées des livres (titre, auteur, etc.)."""
        print("Chargement des items...")
        path = os.path.join(self.data_path, 'items.csv')
        # On gère les erreurs de séparateurs qui arrivent souvent avec les textes de livres
        self.items = pd.read_csv(path, on_bad_lines='skip') 
        print(f"Items chargés : {self.items.shape[0]} livres.")
        return self.items

    def load_submission_sample(self):
        """Charge le fichier d'exemple de soumission pour avoir le format."""
        path = os.path.join(self.data_path, 'sample_submission.csv')
        self.submission_format = pd.read_csv(path)
        return self.submission_format

    def get_all_data(self):
        """Charge tout d'un coup."""
        self.load_interactions()
        self.load_items()
        self.load_submission_sample()
        return self.interactions, self.items, self.submission_format