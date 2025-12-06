# app.py

import os
import sys

import streamlit as st
import pandas as pd

# -------------------------------------------------------------------
# 0. Pour pouvoir réutiliser le code dans src/ (data_loader, recommender)
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_loader import DataLoader
from recommender import ItemItemRecommender, PopularityRecommender


# -------------------------------------------------------------------
# 1. Chargement des données + entraînement des modèles (avec cache)
# -------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Charge les données, entraîne les modèles et renvoie :
    - item_item_model : modèle de recommandation item–item (celui du 0.15488)
    - popular_items   : liste d'items populaires (fallback)
    - interactions    : DataFrame des interactions (u, i, ...)
    - items           : DataFrame des livres (métadonnées)
    - sample          : DataFrame du sample_submission (liste des user_id)
    """
    loader = DataLoader(data_path="data")
    interactions, items, sample = loader.get_all_data()

    # Modèle de popularité (fallback global)
    pop_model = PopularityRecommender(k=50)
    pop_model.fit(interactions, item_col="i")
    popular_items = pop_model.recommend(k=50)

    # Modèle item–item (cosinus + récence)
    item_item_model = ItemItemRecommender(n_recent=5)
    item_item_model.fit(interactions, user_col="u", item_col="i")

    return item_item_model, popular_items, interactions, items, sample


# -------------------------------------------------------------------
# 2. Fonction utilitaire : recommandations formatées pour l'affichage
# -------------------------------------------------------------------
def get_recommendations_for_user(
    model: ItemItemRecommender,
    user_id: int,
    popular_items: list[int],
    items_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """
    Renvoie un DataFrame avec les recommandations pour un user :
    colonnes : [item_id, title, author, subjects] si dispo.
    """
    rec_ids = model.recommend_for_user(
        user_id=user_id,
        k=k,
        popularity_fallback=popular_items,
    )

    if not rec_ids:
        return pd.DataFrame(columns=["item_id", "title"])

    rec_df = pd.DataFrame({"item_id": rec_ids})

    # Harmoniser les colonnes de items.csv avec des noms standard
    # items.csv contient: 'i', 'Title', 'Author', 'Subjects', ...
    items_tmp = items_df.copy().rename(
        columns={
            "i": "item_id",
            "Title": "title",
            "Author": "author",
            "Subjects": "subjects",
        }
    )

    rec_merged = rec_df.merge(items_tmp, on="item_id", how="left")

    display_cols = ["item_id", "title"]
    if "author" in rec_merged.columns:
        display_cols.append("author")
    if "subjects" in rec_merged.columns:
        display_cols.append("subjects")

    return rec_merged[display_cols]


# -------------------------------------------------------------------
# 3. Interface Streamlit
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="📚 Lazy Librarian – Recommender",
        page_icon="📖",
        layout="wide",
    )

    st.title("📚 Lazy Librarian – AI Book Recommender")

    st.markdown(
        """
        Ce démonstrateur utilise un **modèle de filtrage collaboratif item–item**
        basé sur la **similarité cosinus** et la **récence des emprunts**.

        - Baseline officielle : `MAP@10 = 0.15283`  
        - Notre modèle item–item + récence : **≈ 0.15488** sur Kaggle
        """
    )

    with st.spinner("Chargement des données et entraînement du modèle..."):
        model, popular_items, interactions, items, sample = load_models_and_data()

    st.success("Modèle chargé ✅")

    # ----------------------------------------------------------------
    # Sidebar : sélection de l'utilisateur + paramètres
    # ----------------------------------------------------------------
    st.sidebar.header("🔍 Sélection de l'utilisateur")

    user_ids = sorted(sample["user_id"].unique().tolist())
    default_index = 0 if user_ids else 0

    selected_user = st.sidebar.selectbox(
        "Choisir un user_id",
        options=user_ids,
        index=default_index,
        format_func=lambda x: f"User {x}",
    )

    st.sidebar.subheader("⚙️ Paramètres du modèle")
    k = st.sidebar.slider(
        "Nombre de livres recommandés",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.sidebar.markdown(
        """
        💡 Le système recommande des livres :
        - basés sur les **derniers emprunts**  
        - en utilisant un modèle **item–item (cosinus)**  
        - avec un fallback sur la **popularité globale**
        """
    )

    # ----------------------------------------------------------------
    # Onglets : Recommandations / Explorer la collection
    # ----------------------------------------------------------------
    tab_rec, tab_explore = st.tabs(["🎯 Recommandations", "📚 Explorer la collection"])

    # ===== Onglet 1 : Recommandations personnalisées =====
    with tab_rec:
        st.subheader(f"🎯 Recommandations pour l'utilisateur {selected_user}")

        # Statistiques utilisateur
        user_history_full = interactions[interactions["u"] == int(selected_user)]
        nb_events = len(user_history_full)
        nb_items_distincts = (
            user_history_full["i"].nunique() if not user_history_full.empty else 0
        )

        col1, col2 = st.columns(2)
        col1.metric("Nombre total d'emprunts", nb_events)
        col2.metric("Livres distincts empruntés", nb_items_distincts)

        # Bouton pour générer les recommandations
        if st.button("Générer les recommandations"):
            rec_df = get_recommendations_for_user(
                model=model,
                user_id=int(selected_user),
                popular_items=popular_items,
                items_df=items,
                k=k,
            )

            if rec_df.empty:
                st.warning("Pas de recommandation disponible pour cet utilisateur.")
            else:
                st.write(f"Top {k} recommandations :")
                st.dataframe(rec_df, use_container_width=True)

        # Historique récent
        with st.expander("📜 Voir l'historique récent de cet utilisateur"):
            user_history = user_history_full.copy()

            if user_history.empty:
                st.write("Aucun historique trouvé pour cet utilisateur.")
            else:
                st.write(f"Nombre total d'emprunts : {len(user_history)}")
                last_n = 20
                user_history = user_history.tail(last_n)

                # Harmoniser les colonnes avec items
                items_tmp = items.copy().rename(
                    columns={
                        "i": "item_id",
                        "Title": "title",
                        "Author": "author",
                    }
                )

                if "i" in user_history.columns:
                    user_history = user_history.rename(columns={"i": "item_id"})

                hist_merged = user_history.merge(items_tmp, on="item_id", how="left")

                display_cols = ["item_id", "title"]
                if "author" in hist_merged.columns:
                    display_cols.append("author")

                st.dataframe(hist_merged[display_cols], use_container_width=True)

    # ===== Onglet 2 : Explorer la collection =====
    with tab_explore:
        st.subheader("📚 Explorer la collection")

        search = st.text_input("Rechercher un livre (par titre) :")

        items_tmp = items.copy().rename(
            columns={
                "i": "item_id",
                "Title": "title",
                "Author": "author",
                "Subjects": "subjects",
            }
        )

        df_display = items_tmp[["item_id", "title", "author", "subjects"]]

        if search:
            df_display = df_display[
                df_display["title"].str.contains(search, case=False, na=False)
            ]

        st.write(f"{len(df_display)} livres trouvés")
        st.dataframe(df_display.head(200), use_container_width=True)


if __name__ == "__main__":
    main()