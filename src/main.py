from data_loader import DataLoader
from recommender import PopularityRecommender

def main():
    # 1. Chargement
    loader = DataLoader(data_path='data') # ou 'kaggle_data' selon ton dossier réel
    interactions, items, sample_submission = loader.get_all_data()
    
    # 2. Préparation du modèle
    model = PopularityRecommender()
    
    # 3. Entraînement (Le modèle apprend les livres populaires)
    model.fit(interactions)
    
    # 4. Prédiction (On prédit pour tous les utilisateurs du fichier exemple)
    # On récupère la liste des utilisateurs qui attendent une recommandation
    users_to_predict = sample_submission['user_id'].unique()
    
    my_submission = model.predict(users_to_predict)
    
    # 5. Sauvegarde
    print("Sauvegarde du fichier de soumission...")
    my_submission.to_csv('submission_baseline.csv', index=False)
    print("Terminé ! Le fichier 'submission_baseline.csv' est prêt.")

if __name__ == "__main__":
    main()