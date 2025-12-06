# app.py

import os
import sys
import re
from collections import Counter
from typing import List, Dict, Any

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
# 1. Helpers pour sujets & "couvertures"
# -------------------------------------------------------------------
def normalize_items_df(items_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes de items.csv."""
    return items_df.rename(
        columns={
            "i": "item_id",
            "Title": "title",
            "Author": "author",
            "Subjects": "subjects",
        }
    )


def get_cover_emoji(subjects: str | None) -> str:
    """Renvoie une 'couverture' emoji en fonction des sujets."""
    if not isinstance(subjects, str):
        return "📘"
    s = subjects.lower()
    if "psycholog" in s or "psycho" in s:
        return "🧠"
    if "law" in s or "legal" in s or "droit" in s:
        return "⚖️"
    if "computer" in s or "informat" in s or "software" in s:
        return "💻"
    if "medic" in s or "health" in s or "nursing" in s:
        return "🩺"
    if "math" in s or "statistic" in s:
        return "📊"
    if "history" in s or "histor" in s:
        return "🏛️"
    return "📘"


@st.cache_data
def extract_top_subject_tags(items_df: pd.DataFrame, top_n: int = 30) -> List[str]:
    """Construit une liste de tags de sujets les plus fréquents."""
    if "subjects" not in items_df.columns:
        return []

    subjects_series = items_df["subjects"].dropna().astype(str)
    tokens: List[str] = []
    for s in subjects_series:
        for t in re.split(r"[;,/]", s):
            t = t.strip()
            if len(t) > 2:
                tokens.append(t)

    counter = Counter(tokens)
    return [t for t, _ in counter.most_common(top_n)]


# -------------------------------------------------------------------
# 2. Chargement des données + entraînement des modèles (avec cache)
# -------------------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    """
    Charge les données, entraîne les modèles et renvoie :
    - item_item_model : modèle de recommandation item–item (celui du 0.15488)
    - popular_items   : liste d'items populaires (fallback)
    - interactions    : DataFrame des interactions (u, i, ...)
    - items_norm      : DataFrame des livres (colonnes normalisées)
    - sample          : DataFrame du sample_submission (liste des user_id)
    """
    loader = DataLoader(data_path="data")
    interactions, items, sample = loader.get_all_data()

    items_norm = normalize_items_df(items)

    # Modèle de popularité (fallback global)
    pop_model = PopularityRecommender(k=50)
    pop_model.fit(interactions, item_col="i")
    popular_items = pop_model.recommend(k=50)

    # Modèle item–item (cosinus + récence)
    item_item_model = ItemItemRecommender(n_recent=5)
    item_item_model.fit(interactions, user_col="u", item_col="i")

    return item_item_model, popular_items, interactions, items_norm, sample


# -------------------------------------------------------------------
# 3. Fonction utilitaire : recommandations formatées pour l'affichage
# -------------------------------------------------------------------
def get_recommendations_for_user(
    model: ItemItemRecommender,
    user_id: int,
    popular_items: List[int],
    items_df: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """
    Renvoie un DataFrame avec les recommandations pour un user :
    colonnes : [item_id, title, author, subjects].
    """
    rec_ids = model.recommend_for_user(
        user_id=user_id,
        k=k,
        popularity_fallback=popular_items,
    )

    if not rec_ids:
        return pd.DataFrame(columns=["item_id", "title", "author", "subjects"])

    rec_df = pd.DataFrame({"item_id": rec_ids})
    rec_merged = rec_df.merge(items_df, on="item_id", how="left")

    # Ajout d'une colonne "cover" (emoji) pour l'affichage
    rec_merged["cover"] = rec_merged["subjects"].apply(get_cover_emoji)

    display_cols = ["cover", "item_id", "title", "author", "subjects"]
    for col in display_cols:
        if col not in rec_merged.columns:
            rec_merged[col] = None

    return rec_merged[display_cols]


# -------------------------------------------------------------------
# 4. Gestion des favoris (session_state)
# -------------------------------------------------------------------
def init_favorites_state():
    if "favorites" not in st.session_state:
        st.session_state["favorites"] = []  # liste de dicts {item_id, title, author}


def add_favorite(book: Dict[str, Any]):
    init_favorites_state()
    existing_ids = {b["item_id"] for b in st.session_state["favorites"]}
    if book["item_id"] not in existing_ids:
        st.session_state["favorites"].append(book)


def favorites_to_df(items_df: pd.DataFrame) -> pd.DataFrame:
    init_favorites_state()
    if not st.session_state["favorites"]:
        return pd.DataFrame(columns=["item_id", "title", "author", "subjects"])

    fav_df = pd.DataFrame(st.session_state["favorites"]).drop_duplicates("item_id")
    fav_df = fav_df.merge(items_df, on="item_id", how="left", suffixes=("", "_items"))
    if "subjects_items" in fav_df.columns and "subjects" not in fav_df.columns:
        fav_df = fav_df.rename(columns={"subjects_items": "subjects"})
    fav_df["cover"] = fav_df["subjects"].apply(get_cover_emoji)
    display_cols = ["cover", "item_id", "title", "author", "subjects"]
    for col in display_cols:
        if col not in fav_df.columns:
            fav_df[col] = None
    return fav_df[display_cols]


# -------------------------------------------------------------------
# 5. Interface Streamlit
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
        model, popular_items, interactions, items_norm, sample = load_models_and_data()

    st.success("Modèle chargé ✅")
    init_favorites_state()

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
    # Onglets : Recommandations / Explorer la collection / Favoris
    # ----------------------------------------------------------------
    tab_rec, tab_explore, tab_fav = st.tabs(
        ["🎯 Recommandations", "📚 Explorer la collection", "⭐ Ma liste de lecture"]
    )

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
        rec_df = pd.DataFrame()
        if st.button("Générer les recommandations"):
            rec_df = get_recommendations_for_user(
                model=model,
                user_id=int(selected_user),
                popular_items=popular_items,
                items_df=items_norm,
                k=k,
            )

            if rec_df.empty:
                st.warning("Pas de recommandation disponible pour cet utilisateur.")
            else:
                st.write(f"Top {k} recommandations :")
                st.dataframe(rec_df, width="stretch")

        # Section "Ajouter à ma liste" si on a des recos
        if not rec_df.empty:
            st.markdown("### 📝 Ajouter des livres à ma liste de lecture")

            for _, row in rec_df.iterrows():
                cover = row.get("cover", "📘")
                title = row.get("title", "(titre inconnu)")
                author = row.get("author", "(auteur inconnu)")
                item_id = int(row.get("item_id"))

                c1, c2, c3, c4 = st.columns([0.6, 4, 3, 2])
                with c1:
                    st.markdown(f"{cover}")
                with c2:
                    st.markdown(f"**{title}**")
                with c3:
                    st.markdown(f"{author}")
                with c4:
                    if st.button("Ajouter à ma liste", key=f"fav_{selected_user}_{item_id}"):
                        add_favorite(
                            {
                                "item_id": item_id,
                                "title": title,
                                "author": author,
                            }
                        )
                        st.success(f"Ajouté : {title}")

        # Historique récent
        with st.expander("📜 Voir l'historique récent de cet utilisateur"):
            user_history = user_history_full.copy()

            if user_history.empty:
                st.write("Aucun historique trouvé pour cet utilisateur.")
            else:
                st.write(f"Nombre total d'emprunts : {len(user_history)}")
                last_n = 20
                user_history = user_history.tail(last_n)

                if "i" in user_history.columns:
                    user_history = user_history.rename(columns={"i": "item_id"})

                hist_merged = user_history.merge(
                    items_norm, on="item_id", how="left", suffixes=("", "_items")
                )

                display_cols = ["item_id", "title", "author"]
                for col in display_cols:
                    if col not in hist_merged.columns:
                        hist_merged[col] = None

                st.dataframe(hist_merged[display_cols], width="stretch")

    # ===== Onglet 2 : Explorer la collection =====
    with tab_explore:
        st.subheader("📚 Explorer la collection")

        search = st.text_input("Rechercher un livre (par titre) :")

        df_display = items_norm.copy()
        df_display["cover"] = df_display["subjects"].apply(get_cover_emoji)
        display_cols = ["cover", "item_id", "title", "author", "subjects"]
        df_display = df_display[display_cols]

        # Tag cloud & filtres
        top_tags = extract_top_subject_tags(items_norm)
        if top_tags:
            st.markdown("### 🌈 Sujets fréquents")
            st.write(", ".join(top_tags))
            selected_tags = st.multiselect(
                "Filtrer par sujet", options=top_tags, default=[]
            )
        else:
            st.write("Aucun sujet disponible.")
            selected_tags = []

        # Filtre texte sur le titre
        if search:
            df_display = df_display[
                df_display["title"].str.contains(search, case=False, na=False)
            ]

        # Filtre par sujets sélectionnés
        if selected_tags:
            mask = pd.Series(False, index=df_display.index)
            for tag in selected_tags:
                mask = mask | df_display["subjects"].str.contains(
                    tag, case=False, na=False
                )
            df_display = df_display[mask]

        st.write(f"{len(df_display)} livres trouvés")
        st.dataframe(df_display.head(200), width="stretch")

    # ===== Onglet 3 : Ma liste de lecture =====
    with tab_fav:
        st.subheader("⭐ Ma liste de lecture")

        fav_df = favorites_to_df(items_norm)
        if fav_df.empty:
            st.write("Aucun livre dans la liste pour l’instant.")
        else:
            st.dataframe(fav_df, width="stretch")


if __name__ == "__main__":
    main()