import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

st.set_page_config(page_title="MedStat AI Assistant", layout="wide")

# Load CSS safely relative to current file location
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("CSS file not found. UI styling may be limited.")

st.title("🧠 MedStat AI Assistant")
st.markdown('<div class="medico-animation">🩺 MedStat is Ready to Diagnose 📊</div>', unsafe_allow_html=True)

st.markdown("""
A free, offline AI-like assistant for medical students to understand and perform basic statistical analysis on any dataset.
Upload your CSV file, and let the app help you choose the right test, run it, show you graphs, and explain the results in simple terms.
""")

# --- Upload file ---
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.header("Choose Your Analysis")
    analysis_type = st.sidebar.selectbox("Type of Analysis", [
        "Compare Means", "Association Between Categories", "Correlation",
        "Regression", "Check Normality", "Compare Variances"
    ])

    if analysis_type == "Compare Means":
        target = st.sidebar.selectbox("Numeric variable", numeric_cols)
        group = st.sidebar.selectbox("Grouping variable", categorical_cols)
        if target and group:
            groups = df[group].dropna().unique()
            if len(groups) == 2:
                st.info("Recommended test: Independent t-test")
                data1 = df[df[group] == groups[0]][target]
                data2 = df[df[group] == groups[1]][target]
                t_stat, p_val = stats.ttest_ind(data1, data2, nan_policy='omit')
                st.write(f"**t-statistic**: {t_stat:.3f}, **p-value**: {p_val:.3f}")
                st.write("Interpretation: If p < 0.05, the difference is statistically significant.")
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, ax=ax)
                st.pyplot(fig)
            elif len(groups) > 2:
                st.info("Recommended test: One-way ANOVA")
                model = smf.ols(f'{target} ~ C({group})', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, ax=ax)
                st.pyplot(fig)

    elif analysis_type == "Association Between Categories":
        cat1 = st.sidebar.selectbox("First categorical variable", categorical_cols)
        cat2 = st.sidebar.selectbox("Second categorical variable", [c for c in categorical_cols if c != cat1])
        if cat1 and cat2:
            table = pd.crosstab(df[cat1], df[cat2])
            st.write("Contingency Table:", table)
            chi2, p, _, _ = stats.chi2_contingency(table)
            st.write(f"**Chi-square**: {chi2:.3f}, **p-value**: {p:.3f}")
            st.write("Interpretation: If p < 0.05, the association is significant.")

    elif analysis_type == "Correlation":
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2", [col for col in numeric_cols if col != x])
        if x and y:
            r, p = stats.pearsonr(df[x], df[y])
            st.write(f"**Pearson r**: {r:.3f}, **p-value**: {p:.3f}")
            fig, ax = plt.subplots()
            sns.scatterplot(x=x, y=y, data=df, ax=ax)
            st.pyplot(fig)

    elif analysis_type == "Regression":
        y = st.sidebar.selectbox("Dependent variable", numeric_cols)
        x = st.sidebar.selectbox("Independent variable", [col for col in numeric_cols if col != y])
        if x and y:
            model = smf.ols(f'{y} ~ {x}', data=df).fit()
            st.write(model.summary())
            fig, ax = plt.subplots()
            sns.regplot(x=x, y=y, data=df, ax=ax)
            st.pyplot(fig)

    elif analysis_type == "Check Normality":
        col = st.sidebar.selectbox("Numeric column", numeric_cols)
        if col:
            stat, p = stats.shapiro(df[col].dropna())
            st.write(f"**W-statistic**: {stat:.3f}, **p-value**: {p:.3f}")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    elif analysis_type == "Compare Variances":
        target = st.sidebar.selectbox("Numeric variable", numeric_cols)
        group = st.sidebar.selectbox("Grouping variable", categorical_cols)
        if target and group:
            groups = df[group].dropna().unique()
            if len(groups) == 2:
                g1 = df[df[group] == groups[0]][target]
                g2 = df[df[group] == groups[1]][target]
                stat, p = stats.levene(g1, g2)
                st.write(f"**Levene's stat**: {stat:.3f}, **p-value**: {p:.3f}")
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, ax=ax)
                st.pyplot(fig)

else:
    st.info("Please upload a CSV file to get started.")
