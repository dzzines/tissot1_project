import pandas as pd

class PopularityRecommender:
    """
    Un système de recommandation simple qui suggère toujours 
    les livres les plus populaires, peu importe l'utilisateur.
    """
    
    def __init__(self):
        self.top_items = []
        
    def fit(self, interactions_df):
        """
        Apprend quels sont les livres les plus populaires.
        :param interactions_df: Le DataFrame des interactions (u, i, t)
        """
        print("Entraînement du modèle de popularité...")
        
        # On compte combien de fois chaque livre (i) apparaît
        popularity_counts = interactions_df['i'].value_counts()
        
        # On garde seulement les 10 premiers (les IDs des livres)
        self.top_items = popularity_counts.head(10).index.tolist()
        
        print(f"Top 10 livres identifiés : {self.top_items}")
        
    def predict(self, user_ids):
        """
        Génère les recommandations pour une liste d'utilisateurs.
        :param user_ids: Liste des IDs utilisateurs à qui recommander
        :return: DataFrame au format attendu par Kaggle
        """
        # On transforme la liste [123, 456] en chaîne de caractères "123 456"
        # car c'est le format demandé par Kaggle (espace entre les IDs)
        prediction_string = " ".join(map(str, self.top_items))
        
        predictions = []
        for user in user_ids:
            predictions.append({
                'user_id': user, 
                'recommendation': prediction_string
            })
            
        return pd.DataFrame(predictions)