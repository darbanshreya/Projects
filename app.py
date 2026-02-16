import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Design2Wear-AI", layout="wide")

# Find Kaggle images
import kagglehub
IMG_PATH = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
IMG_FOLDER = os.path.join(IMG_PATH, 'images')

# ====================================
# LOAD FULL 44K DATASET
# ====================================
@st.cache_data
def load_data():
    df = pd.read_csv('fashion_dataset_44k_clean.csv')
    BAD = ['Briefs','Boxers','Bra','Innerwear Vests','Trunk','Socks',
           'Caps','Ties','Belts','Wallets','Camisoles','Shapewear',
           'Stockings','Tights','Bath Robe','Baby Dolls','Swimwear',
           'Pendant','Earrings','Ring','Bracelet','Necklace And Chains',
           'Nail Polish','Lipstick','Perfume And Body Mist','Deodorant',
           'Sunglasses','Flip Flops','Sandals','Sports Sandals']
    df = df[~df['article_type'].isin(BAD)].copy()
    return df

df = load_data()

# ====================================
# HEADER
# ====================================
st.markdown(
    '<div style="text-align:center;padding:10px 0 20px">'
    '<h1 style="font-size:42px;margin:0">Design2Wear<span style="color:#6366f1">AI</span></h1>'
    '<p style="color:#888">Content-based recommendation system | '
    'TensorFlow + Keras + scikit-learn | <b>{:,} products</b></p></div>'.format(len(df)),
    unsafe_allow_html=True
)

# ====================================
# SIDEBAR
# ====================================
st.sidebar.header("Your Preferences")

preset = st.sidebar.selectbox("Quick Presets", [
    "Custom",
    "Wedding (Women)", "Office (Men)", "Night Out (Women)",
    "Casual (Men)", "Festival (Women)", "Gym (Men)"
])

presets_map = {
    "Wedding (Women)": {"color":"red","style":"ethnic","occasion":"wedding","season":"fall","fabric":"silk","size":"M","gender":"women","skin_tone":"medium"},
    "Office (Men)": {"color":"blue","style":"formal","occasion":"office","season":"summer","fabric":"cotton","size":"L","gender":"men","skin_tone":"fair"},
    "Night Out (Women)": {"color":"black","style":"glamorous","occasion":"party","season":"winter","fabric":"satin","size":"S","gender":"women","skin_tone":"dark"},
    "Casual (Men)": {"color":"white","style":"casual","occasion":"daily_wear","season":"summer","fabric":"cotton","size":"M","gender":"men","skin_tone":"medium"},
    "Festival (Women)": {"color":"pink","style":"ethnic","occasion":"festival","season":"spring","fabric":"silk","size":"M","gender":"women","skin_tone":"tan"},
    "Gym (Men)": {"color":"grey","style":"sporty","occasion":"gym","season":"winter","fabric":"cotton","size":"L","gender":"men","skin_tone":"light"},
}

if preset != "Custom" and preset in presets_map:
    d = presets_map[preset]
else:
    d = {"color":"red","style":"ethnic","occasion":"wedding","season":"fall","fabric":"silk","size":"M","gender":"women","skin_tone":"medium"}

colors_list = sorted(df['color'].dropna().unique().tolist())
styles_list = sorted(df['style'].dropna().unique().tolist())
occasions_list = sorted(df['occasion'].dropna().unique().tolist())
seasons_list = sorted(df['season'].dropna().unique().tolist())
fabrics_list = sorted(df['fabric'].dropna().unique().tolist())
skin_tones_list = sorted(df['skin_tone'].dropna().unique().tolist())

def safe_idx(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return 0

gender = st.sidebar.selectbox("Gender", ["women", "men"], index=0 if d["gender"]=="women" else 1)
color = st.sidebar.selectbox("Color", colors_list, index=safe_idx(colors_list, d["color"]))
style = st.sidebar.selectbox("Style", styles_list, index=safe_idx(styles_list, d["style"]))
occasion = st.sidebar.selectbox("Occasion", occasions_list, index=safe_idx(occasions_list, d["occasion"]))
season = st.sidebar.selectbox("Season", seasons_list, index=safe_idx(seasons_list, d["season"]))
fabric = st.sidebar.selectbox("Fabric", fabrics_list, index=safe_idx(fabrics_list, d["fabric"]))
size = st.sidebar.selectbox("Size", ['XS','S','M','L','XL','XXL'], index=['XS','S','M','L','XL','XXL'].index(d["size"]))
skin_tone = st.sidebar.selectbox("Skin Tone", skin_tones_list, index=safe_idx(skin_tones_list, d["skin_tone"]))
n_results = st.sidebar.slider("Number of results", 4, 20, 8)

# ====================================
# RECOMMENDATION ENGINE
# ====================================
if st.sidebar.button("Get Recommendations", type="primary", use_container_width=True):

    # STEP 1: HARD FILTER on gender
    df_filtered = df[df['gender'] == gender].copy()

    if len(df_filtered) == 0:
        st.error(f"No {gender} outfits found!")
    else:
        st.info(f"Searching **{len(df_filtered):,}** {gender}'s outfits from **{len(df):,}** total products...")

        # STEP 2: Encode features
        feature_cols = ['article_type', 'color', 'style', 'occasion',
                        'season', 'fabric', 'size', 'skin_tone']

        user_row = pd.DataFrame([{
            'article_type': 'any',
            'color': color,
            'style': style,
            'occasion': occasion,
            'season': season,
            'fabric': fabric,
            'size': size,
            'skin_tone': skin_tone,
        }])

        combined = pd.concat([df_filtered[feature_cols], user_row], ignore_index=True)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = ohe.fit_transform(combined[feature_cols])

        user_vector = encoded[-1:]
        outfit_vectors = encoded[:-1]

        # STEP 3: Cosine similarity
        sims = cosine_similarity(user_vector, outfit_vectors)[0]

        df_results = df_filtered.copy()
        df_results['match_score'] = sims

        # STEP 4: Sort, deduplicate (so we don't show 8 identical blue shirts)
        df_results = df_results.sort_values('match_score', ascending=False)
        df_results['dedup_key'] = df_results['article_type'] + '_' + df_results['color']
        df_results = df_results.drop_duplicates(subset='dedup_key', keep='first')
        df_results = df_results.head(n_results)

        # ====================================
        # DISPLAY RESULTS
        # ====================================
        st.markdown(f"### Showing **{gender.upper()}** only")
        st.caption(f"Searched {len(df_filtered):,} {gender}'s products | Ranked by cosine similarity")

        cols_per_row = 4
        for row_start in range(0, len(df_results), cols_per_row):
            row_df = df_results.iloc[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for idx, (_, outfit) in enumerate(row_df.iterrows()):
                with cols[idx]:
                    score = outfit['match_score'] * 100

                    # Product image
                    img_path = os.path.join(IMG_FOLDER, f"{outfit['outfit_id']}.jpg")
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.markdown("No image available")

                    # Score
                    if score >= 60:
                        st.success(f"**{score:.0f}% match** — TOP PICK")
                    elif score >= 35:
                        st.warning(f"**{score:.0f}% match** — GOOD")
                    else:
                        st.info(f"**{score:.0f}% match**")

                    # Details
                    st.markdown(f"**{outfit['article_type']}**")
                    st.caption(f"{outfit['color']} | {outfit['fabric']} | {outfit['season']}")

                    # Matched attributes
                    matched = []
                    if str(outfit['color']).strip() == color.strip():
                        matched.append("color")
                    if str(outfit['style']).strip() == style.strip():
                        matched.append("style")
                    if str(outfit['occasion']).strip() == occasion.strip():
                        matched.append("occasion")
                    if str(outfit['season']).strip() == season.strip():
                        matched.append("season")
                    if str(outfit['fabric']).strip() == fabric.strip():
                        matched.append("fabric")
                    if str(outfit['size']).strip() == size.strip():
                        matched.append("size")
                    if str(outfit['skin_tone']).strip() == skin_tone.strip():
                        matched.append("skin tone")
                    matched.append("gender")

                    st.markdown(f"**{len(matched)}/8 matched:** {', '.join(matched)}")

        st.markdown("---")
        top_score = df_results['match_score'].iloc[0] * 100
        avg_score = df_results['match_score'].mean() * 100
        st.caption(f"Top: {top_score:.0f}% | Avg: {avg_score:.0f}% | Products: {len(df_filtered):,} | Engine: Cosine Similarity")

else:
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#888">'
        '<h3>Select preferences in the sidebar, then click Get Recommendations</h3>'
        '</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
c = st.columns(6)
for i, t in enumerate(['Python','TensorFlow','Keras','scikit-learn','Pandas','Streamlit']):
    c[i].markdown(f"<div style='text-align:center;color:#6366f1;font-weight:700;font-size:12px'>{t}</div>", unsafe_allow_html=True)
