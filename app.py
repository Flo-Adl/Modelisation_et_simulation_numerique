import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# CONFIG PAGE
# =========================================================
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("Maintenance Prédictive : Turbofan Aircraft Engine")
st.markdown("Estimation du **RUL (Remaining Useful Life)** : Données → IA → Optimisation")
st.markdown("<br>", unsafe_allow_html=True)

# ========================================================= 
# SIDEBAR 
# ========================================================= 
st.sidebar.header("Paramètres") 
st.sidebar.markdown("<br>", unsafe_allow_html=True) 

# --------------------------------------------------------- 
# SOURCE DES DONNÉES 
# --------------------------------------------------------- 
data_source = st.sidebar.radio( "Source des données", [ "Données synthétiques", "NASA CMAPSS (fichiers locaux)" ] ) 

# ========================================================= 
# PARAMÈTRES DATASET 
# ========================================================= 
N_ENGINES = st.sidebar.slider( "Nombre de moteurs", 10, 200, 50 ) 

if data_source == "Données synthétiques": 
    MAX_CYCLES = st.sidebar.slider( "Cycles max", 100, 500, 200 ) 

else: 
    dataset_choice = st.sidebar.selectbox( "Dataset NASA", ["FD001", "FD002", "FD003", "FD004"] ) 
    MAX_CYCLES = None 
    
st.sidebar.markdown(2*"<br>", unsafe_allow_html=True) 
RUL_CRIT = st.sidebar.slider( "RUL critique", 10, 100, 30 ) 
K_MAINT = st.sidebar.slider( "Capacité atelier", 1, 10, 3 ) 
N_MIN = st.sidebar.slider( "Avions minimum disponibles", 1, 20, 5 )

# Hyperparamètres ML fixés (standards)
TEST_SIZE = 0.20
K_FOLDS   = 5

np.random.seed(42)

# =========================================================
# COLONNES NASA CMAPSS
# =========================================================
CMAPSS_COLS = [
    "engine_id", "cycle",
    "op1", "op2", "op3",                          # conditions opérationnelles
    "s1","s2","s3","s4","s5","s6","s7",
    "s8","s9","s10","s11","s12","s13","s14",
    "s15","s16","s17","s18","s19","s20","s21"     # 21 capteurs
]

# Les 4 capteurs retenus dans notre modèle, mappés sur les capteurs CMAPSS
# s2  = température sortie fan (proxy EGT)
# s4  = pression sortie compresseur (proxy Pratio)
# s11 = température sortie compresseur (proxy Vfan par corrélation)
# s17 = rendement compresseur (proxy SFC)
CMAPSS_SENSOR_MAP = {
    "EGT":    "s2",
    "Pratio": "s4",
    "Vfan":   "s11",
    "SFC":    "s17",
}

# =========================================================
# CHARGEMENT DES DONNÉES
# =========================================================

def load_cmapss(train_path, rul_path):
    # Chargement fichier d'entraînement
    df = pd.read_csv(train_path, sep=r"\s+", header=None, names=CMAPSS_COLS)
    df.dropna(axis=1, how="all", inplace=True)  # supprime colonnes vides en fin de ligne

    # Calcul du RUL : pour chaque moteur, RUL = nb_cycles_total - cycle_courant
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df = df.join(max_cycle.rename("max_cycle"), on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    # Renommage des capteurs pour correspondre à notre modèle
    for feature_name, sensor_col in CMAPSS_SENSOR_MAP.items():
        df[feature_name] = df[sensor_col]
    return df

def generate_engine_data(engine_id, max_cycles):
    life   = np.random.randint(80, max_cycles)
    cycles = np.arange(1, life + 1)
    rul    = life - cycles

    # Palier initial : dégradation commence à 20% de la vie
    degradation_start = int(0.20 * life)
    ramp = np.zeros(life)
    ramp[degradation_start:] = np.arange(life - degradation_start)

    egt = 600 + 0.8   * ramp + np.random.normal(0, 5, life)
    sfc = 0.3 + 0.002 * ramp + 0.0003 * (egt - 600) + np.random.normal(0, 0.01, life)
    vfan = 0.5 + 0.01  * ramp + np.abs(np.random.normal(0, 0.05, life))
    pratio = 30 - 0.05  * ramp + np.random.normal(0, 0.3, life)

    df = pd.DataFrame({
        "engine_id": engine_id,
        "cycle": cycles,
        "RUL": rul,
        "EGT": egt,
        "Vfan": vfan,
        "Pratio": pratio,
        "SFC": sfc,
    })

    for col in ["EGT", "Vfan", "Pratio", "SFC"]:
        mask = np.random.rand(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    return df

# --- Chargement selon le mode choisi ---
if data_source == "NASA CMAPSS (fichiers locaux)":
    train_path = f"data/train_{dataset_choice}.txt"
    rul_path   = f"data/RUL_{dataset_choice}.txt"

    try:
        data = load_cmapss(train_path, rul_path)
        # On limite à N_ENGINES moteurs pour rester cohérent avec le slider
        engines_available = sorted(data["engine_id"].unique())
        engines_kept = engines_available[:N_ENGINES]
        data = data[data["engine_id"].isin(engines_kept)].copy()

    except FileNotFoundError:
        st.stop()

else:
    # Données synthétiques
    dfs  = [generate_engine_data(i, MAX_CYCLES) for i in range(N_ENGINES)]
    data = pd.concat(dfs, ignore_index=True)
    data[["EGT","Vfan","Pratio","SFC"]] = (
        data.groupby("engine_id")[["EGT","Vfan","Pratio","SFC"]]
        .transform(lambda s: s.interpolate().ffill().bfill())
    )

# =========================================================
# MACHINE LEARNING
# =========================================================
FEATURES = ["cycle", "EGT", "Vfan", "Pratio", "SFC"]
X = data[FEATURES]
y = data["RUL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Régression Linéaire
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = np.clip(lr.predict(X_test_sc), 0, None)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = np.clip(rf.predict(X_test), 0, None)

# Métriques
def compute_metrics(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
    }

metrics_lr = compute_metrics(y_test, y_pred_lr)
metrics_rf = compute_metrics(y_test, y_pred_rf)

cv_lr = cross_val_score(LinearRegression(), X_train_sc, y_train, cv=K_FOLDS, scoring="r2")
cv_rf = cross_val_score(rf, X_train, y_train, cv=K_FOLDS, scoring="r2")

# =========================================================
# ONGLETS
# =========================================================
tab1, tab2, tab3, tab4, tab5 ,tab6= st.tabs([
    "KPIs et Données",
    "Modeles et Comparaison",
    "Visualisation Capteurs",
    "Contraintes et Optimisation",
    "Validation",
    "Objectifs",
])

# ─────────────────────────────────────────────
# TAB 1 — KPIs et Données
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Métriques du meilleur modèle (Random Forest)")
    st.caption("MAE = erreur absolue moyenne | RMSE = erreur quadratique | R2 = qualité d'ajustement global")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics_rf['MAE']:.2f}")
    col2.metric("RMSE", f"{metrics_rf['RMSE']:.2f}")
    col3.metric("R2", f"{metrics_rf['R2']:.3f}")
    col4.metric("Observations", f"{len(data):,}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Dataset")
    if data_source == "NASA CMAPSS (fichiers locaux)":
        st.caption(f"Source : NASA CMAPSS : {dataset_choice}")
    else:
        st.caption("Source : données synthétiques. NaN imputés par interpolation linéaire.")
    st.dataframe(data[["engine_id","cycle","RUL","EGT","Vfan","Pratio","SFC"]].head(len(data)),
                 use_container_width=True)

    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger dataset CSV", csv, "engines_dataset.csv", "text/csv")

# ─────────────────────────────────────────────
# TAB 2 — Modèles et Comparaison
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Comparaison : Régression Linéaire vs Random Forest")

    df_compare = pd.DataFrame({
        "Métrique": ["MAE", "RMSE", "R2",f"R2 CV ({K_FOLDS}-fold) moy.", "R2 CV std"],
        "Régression Lin.": [f"{metrics_lr['MAE']:.2f}", f"{metrics_lr['RMSE']:.2f}", f"{metrics_lr['R2']:.3f}",f"{cv_lr.mean():.3f}", f"{cv_lr.std():.3f}"],
        "Random Forest": [f"{metrics_rf['MAE']:.2f}", f"{metrics_rf['RMSE']:.2f}",f"{metrics_rf['R2']:.3f}",f"{cv_rf.mean():.3f}", f"{cv_rf.std():.3f}"]})
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Régression Linéaire** : RUL prédit vs réel")
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=y_test, y=y_pred_lr, mode="markers",
                                    marker=dict(size=4, opacity=0.5), name="LR"))
        fig_lr.add_trace(go.Scatter(x=[0, int(y_test.max())], y=[0, int(y_test.max())],
                                    mode="lines", line=dict(dash="dash", color="red"),
                                    name="Parfait"))
        fig_lr.update_layout(xaxis_title="RUL réel", yaxis_title="RUL prédit", height=400)
        st.plotly_chart(fig_lr, use_container_width=True)

    with col_b:
        st.markdown("**Random Forest** : RUL prédit vs réel")
        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(x=y_test, y=y_pred_rf, mode="markers",
                                    marker=dict(size=4, opacity=0.5, color="green"), name="RF"))
        fig_rf.add_trace(go.Scatter(x=[0, int(y_test.max())], y=[0, int(y_test.max())],
                                    mode="lines", line=dict(dash="dash", color="red"),
                                    name="Parfait"))
        fig_rf.update_layout(xaxis_title="RUL réel", yaxis_title="RUL prédit", height=400)
        st.plotly_chart(fig_rf, use_container_width=True)

    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig_imp = px.bar(importances, orientation="h",
                     labels={"value": "Importance", "index": "Capteur"},
                     title="Feature Importance")
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption("Indique quels capteurs contribuent le plus à la prédiction du RUL.")

    residuals = y_test - y_pred_rf
    fig_res = px.histogram(residuals, nbins=40,
                           title="Résidus = RUL réel - RUL prédit (RF)")
    fig_res.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_res, use_container_width=True)
    st.caption("Un bon modèle a des résidus centrés sur 0 et symétriques. "
            "Un biais systématique indique une erreur de modélisation.")

# ─────────────────────────────────────────────
# TAB 3 — Visualisation Capteurs
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Evolution des capteurs & RUL")
    if data_source == "Données synthétiques":
        st.info("Palier initial : les capteurs restent stables pendant ~20% de la vie "
                "du moteur avant que la dégradation ne s'amorce.")

    engine_sel = st.selectbox("Choisir un moteur", sorted(data["engine_id"].unique()))
    engine_df  = data[data["engine_id"] == engine_sel].copy()
    engine_df["RUL_pred_RF"] = np.clip(rf.predict(engine_df[FEATURES]), 0, None)

    sensor = st.selectbox("Capteur à afficher", ["EGT", "Vfan", "Pratio", "SFC"])
    fig_s  = px.line(engine_df, x="cycle", y=sensor,
                     title=f"{sensor} — Moteur {engine_sel}")
    st.plotly_chart(fig_s, use_container_width=True)

    fig_rul = go.Figure()
    fig_rul.add_trace(go.Scatter(x=engine_df["cycle"], y=engine_df["RUL"],
                                 mode="lines", name="RUL réel",
                                 line=dict(color="steelblue")))
    fig_rul.add_trace(go.Scatter(x=engine_df["cycle"], y=engine_df["RUL_pred_RF"],
                                 mode="lines", name="RUL prédit (RF)",
                                 line=dict(color="green", dash="dash")))
    fig_rul.add_hline(y=RUL_CRIT, line_dash="dot", line_color="orange",
                      annotation_text=f"Seuil critique g1 = {RUL_CRIT} cycles")
    fig_rul.add_hrect(y0=0, y1=RUL_CRIT, fillcolor="red", opacity=0.07,
                      annotation_text="Zone danger")
    fig_rul.update_layout(xaxis_title="Cycle", yaxis_title="RUL (cycles)",
                          title=f"Evolution RUL — Moteur {engine_sel}")
    st.plotly_chart(fig_rul, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 4 — Contraintes & Optimisation
# ─────────────────────────────────────────────
with tab4:
    st.subheader("Contraintes du problème d'optimisation")
    st.info(
        "- **g1** : Maintenance obligatoire si RUL <= RUL_crit (sécurité EASA/FAA)\n"
        "- **g2** : Nombre d'interventions simultanées <= capacité atelier K_maint\n"
        "- **g3** : Nombre d'avions disponibles >= N_min à tout instant"
    )

    latest = data.groupby("engine_id").tail(1).copy()
    latest["RUL_pred"] = np.clip(rf.predict(latest[FEATURES]), 0, None)

    # g1 — sécurité
    latest["g1_violée"] = latest["RUL_pred"] <= RUL_CRIT

    # Variable de décision xi
    # Maintenance urgente si g1 violée, sinon préventive à RUL_pred/2
    latest["xi"] = np.where(
        latest["g1_violée"],
        latest["cycle"] + 1,
        latest["cycle"] + (latest["RUL_pred"] / 2).astype(int)
    )

    # g2 — capacité atelier
    slot_counts = latest["xi"].value_counts().rename("nb_interventions")
    latest = latest.join(slot_counts, on="xi")
    latest["g2_violée"] = latest["nb_interventions"] > K_MAINT

    # g3 — disponibilité flotte
    total_engines = len(latest)
    min_dispo = total_engines - slot_counts.max() if len(slot_counts) > 0 else total_engines
    g3_ok = min_dispo >= N_MIN

    n_g1 = latest["g1_violée"].sum()
    n_g2 = latest["g2_violée"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("g1 — Moteurs critiques", f"{n_g1} / {total_engines}")
    c2.metric("g2 — Surcharge atelier", f"{slot_counts.max() if len(slot_counts) else 0} / {K_MAINT} max")
    c3.metric("g3 — Dispo. minimale flotte", f"{min_dispo} avions")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Plan de maintenance optimal xi")
    st.caption(
        "xi = cycle auquel intervenir sur le moteur i. "
        "Maintenance urgente si g1 violée, sinon préventive à RUL_pred/2."
    )

    plan = latest[["engine_id","cycle","RUL_pred","xi","g1_violée","g2_violée"]].copy()
    plan.columns = ["Moteur","Cycle actuel","RUL prédit",
                    "xi (intervenir au cycle)","g1 violée","g2 violée"]
    plan["Statut"] = np.where(
        plan["g1 violée"], "CRITIQUE",
        np.where(plan["g2 violée"], "SURCHARGE", "OK")
    )
    st.dataframe(plan, use_container_width=True, hide_index=True)

    fig_plan = px.scatter(
        plan,
        x="Moteur", y="xi (intervenir au cycle)",
        color="Statut",
        color_discrete_map={"CRITIQUE": "red", "SURCHARGE": "orange", "OK": "green"},
        title="Plan de maintenance xi par moteur",
        labels={"xi (intervenir au cycle)": "Cycle d'intervention xi"},
    )
    st.plotly_chart(fig_plan, use_container_width=True)

# ─────────────────────────────────────────────
# TAB 5 — Validation & Critique
# ─────────────────────────────────────────────
with tab5:
    st.subheader("Validation et critique de la solution")

    st.markdown("### 1. Cohérence physique")
    st.caption("Le RUL doit être positif, décroissant, et s'annuler à la défaillance.")
    rul_neg = (latest["RUL_pred"] < 0).sum()
    st.write(f"- Prédictions négatives (invalides) : **{rul_neg}** (clippées à 0)")
    st.write(f"- RUL moyen prédit : **{latest['RUL_pred'].mean():.1f} cycles**")
    st.write(f"- RUL médian prédit : **{latest['RUL_pred'].median():.1f} cycles**")

    st.markdown("### 2. Analyse de sensibilité")
    st.caption(
        "On perturbe les données d'entrée (+x% sur EGT) et on observe l'impact sur le RUL prédit. "
        "Un problème bien conditionné est peu sensible aux petites perturbations."
    )

    perturb_pct = st.slider("Perturbation EGT (%)", 1, 100, 5)
    X_test_perturbed = X_test.copy()
    X_test_perturbed["EGT"] *= (1 + perturb_pct / 100)
    y_pred_rf_perturbed = np.clip(rf.predict(X_test_perturbed), 0, None)
    delta_rul = np.abs(y_pred_rf - y_pred_rf_perturbed).mean()

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Variation moyenne du RUL prédit", f"{delta_rul:.2f} cycles")
    col_s2.metric("Sensibilité relative", f"{delta_rul / y_pred_rf.mean() * 100:.1f}%")

    st.markdown("### 3. Vérification des contraintes")
    st.caption("Toutes les contraintes doivent être satisfaites pour que x* soit admissible.")
    st.write(f"- **g1** (RUL critique) : {n_g1} moteur(s) critique(s) — " + ("Maintenance urgente à planifier" if n_g1 > 0 else "Satisfaite"))
    st.write(f"- **g2** (capacité atelier) : max {slot_counts.max() if len(slot_counts) else 0} " f"interventions simultanées / {K_MAINT} max — " + ("Surcharge détectée" if latest['g2_violée'].any() else "Satisfaite"))
    st.write(f"- **g3** (dispo. flotte) : {min_dispo} avions disponibles / {N_MIN} requis — " + ("Satisfaite" if g3_ok else "Non satisfaite"))
    
# ─────────────────────────────────────────────

# TAB 6 — Fonction Objectif

# ─────────────────────────────────────────────

with tab6:
    st.subheader("Formulation de la fonction objectif")
    st.info("""
    ```
    Dans notre contexte industriel, l'objectif est de minimiser le coût global de la flotte
    tout en garantissant le respect absolu des contraintes de sécurité.

    La fonction objectif représente mathématiquement le compromis entre :

    - le coût des maintenances programmées ;
    - le coût espéré des défaillances moteur.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # PARAMETRES ECONOMIQUES
    # =====================================================

    st.markdown("### Paramètres économiques")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        C_prog = st.slider("Coût maintenance programmée C_prog",1000,100000,20000,step=1000)

    with col_c2:
        C_panne = st.slider("Coût panne moteur C_panne",10000,1000000,250000,step=10000)

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # PROBABILITE DE PANNE
    # =====================================================

    latest_obj = latest.copy()
    latest_obj["P_panne"] = np.clip(1 - (latest_obj["RUL_pred"] / latest_obj["RUL_pred"].max()),0,1)
    latest_obj["C_planifie"] = np.where(latest_obj["RUL_pred"] <= RUL_CRIT,C_prog,0.5 * C_prog)
    latest_obj["C_panne_esp"] = (C_panne * latest_obj["P_panne"])
    latest_obj["C_total"] = (latest_obj["C_planifie"]+ latest_obj["C_panne_esp"])

    # =====================================================
    # KPI ECONOMIQUES
    # =====================================================

    st.markdown("### Analyse économique")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Coût maintenance",f"{latest_obj['C_planifie'].sum():,.0f} €")
    k2.metric("Coût panne espéré",f"{latest_obj['C_panne_esp'].sum():,.0f} €")
    k3.metric("Coût total f(x)",f"{latest_obj['C_total'].sum():,.0f} €")
    k4.metric("Probabilité panne moyenne",f"{100 * latest_obj['P_panne'].mean():.1f} %")
    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # VISUALISATION COUTS
    # =====================================================

    st.markdown("### Répartition des coûts")

    fig_cost = px.bar(latest_obj,x="engine_id",
                y=["C_planifie", "C_panne_esp"],
                title="Décomposition des coûts par moteur",
                labels={"value": "Coût (€)","engine_id": "Moteur"
                },barmode="stack")
    st.plotly_chart(fig_cost, use_container_width=True)

    # =====================================================
    # TENSION FONDAMENTALE
    # =====================================================
    st.markdown("Compromis maintenance / risque")

    st.info(
    "Intervenir tôt réduit le risque de panne mais augmente "
    "les coûts de maintenance. "
    "Intervenir tard réduit les coûts immédiats mais augmente "
    "le risque financier associé aux défaillances moteur."
    )

    fig_tradeoff = go.Figure()
    x_trade = np.linspace(0, 100, 100)

    maintenance_cost = 100 - x_trade
    failure_cost = x_trade**1.5 / 10

    total_cost = maintenance_cost + failure_cost

    fig_tradeoff.add_trace(go.Scatter(x=x_trade,y=maintenance_cost,mode="lines",name="Coût maintenance"))
    fig_tradeoff.add_trace(go.Scatter(x=x_trade,y=failure_cost,mode="lines",name="Coût panne"))
    fig_tradeoff.add_trace(go.Scatter(x=x_trade,y=total_cost,mode="lines",name="Coût total f(x)",line=dict(width=4)))
    fig_tradeoff.update_layout(xaxis_title="Temps avant intervention",yaxis_title="Coût",height=500)

    st.plotly_chart(fig_tradeoff, use_container_width=True)

    # =====================================================
    # TABLEAU DETAILLE
    # =====================================================

    st.markdown("### Détail économique par moteur")
    st.dataframe(latest_obj[["engine_id","RUL_pred","P_panne","C_planifie","C_panne_esp","C_total"]],use_container_width=True)
        