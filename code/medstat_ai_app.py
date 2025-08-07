import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

st.set_page_config(page_title="MedStat AI Assistant", layout="wide")
import os

css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 MedStat AI Assistant")
# Inject CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Animation
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

    # Detect variable types
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.header("Choose Your Analysis")
    analysis_type = st.sidebar.selectbox("Type of Analysis", [
        "Compare Means", "Association Between Categories", "Correlation", "Regression", "Check Normality", "Compare Variances"])

    st.sidebar.write("---")

    # Handle each analysis type
    if analysis_type == "Compare Means":
        st.subheader("Compare Means")
        target = st.sidebar.selectbox("Choose numeric variable to compare", numeric_cols)
        group = st.sidebar.selectbox("Choose grouping variable", categorical_cols)

        if target and group:
            unique_groups = df[group].dropna().unique()
            if len(unique_groups) == 2:
                st.info("Recommended test: Independent t-test")
                group1, group2 = unique_groups[0], unique_groups[1]
                data1 = df[df[group] == group1][target]
                data2 = df[df[group] == group2][target]
                t_stat, p_val = stats.ttest_ind(data1, data2, nan_policy='omit')
                st.write(f"**t-statistic**: {t_stat:.3f}, **p-value**: {p_val:.3f}")
                st.write("Interpretation: If p < 0.05, the difference between group means is statistically significant.")
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, ax=ax)
                st.pyplot(fig)
            elif len(unique_groups) > 2:
                st.info("Recommended test: One-way ANOVA")
                model = smf.ols(f'{target} ~ C({group})', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                st.write(anova_table)
                st.write("Interpretation: If p < 0.05, at least one group mean is significantly different.")
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, ax=ax)
                st.pyplot(fig)

    elif analysis_type == "Association Between Categories":
        st.subheader("Association Between Categorical Variables")
        cat1 = st.sidebar.selectbox("First categorical variable", categorical_cols)
        cat2 = st.sidebar.selectbox("Second categorical variable", [c for c in categorical_cols if c != cat1])

        if cat1 and cat2:
            contingency = pd.crosstab(df[cat1], df[cat2])
            st.write("Contingency Table:")
            st.write(contingency)
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            st.write(f"**Chi-square statistic**: {chi2:.3f}, **p-value**: {p:.3f}")
            st.write("Interpretation: If p < 0.05, there is a significant association between the variables.")

    elif analysis_type == "Correlation":
        st.subheader("Correlation Between Two Numeric Variables")
        x = st.sidebar.selectbox("Variable 1", numeric_cols)
        y = st.sidebar.selectbox("Variable 2", [col for col in numeric_cols if col != x])

        if x and y:
            r, p = stats.pearsonr(df[x], df[y])
            st.write(f"**Pearson correlation coefficient**: {r:.3f}, **p-value**: {p:.3f}")
            st.write("Interpretation: Correlation ranges from -1 to 1. If p < 0.05, the correlation is significant.")
            fig, ax = plt.subplots()
            sns.scatterplot(x=x, y=y, data=df, ax=ax)
            st.pyplot(fig)

    elif analysis_type == "Regression":
        st.subheader("Linear Regression")
        y = st.sidebar.selectbox("Dependent variable", numeric_cols)
        x = st.sidebar.selectbox("Independent variable", [col for col in numeric_cols if col != y])

        if x and y:
            model = smf.ols(f'{y} ~ {x}', data=df).fit()
            st.write(model.summary())
            st.write("Interpretation: Look at the p-values of coefficients and R-squared for model fit.")
            fig, ax = plt.subplots()
            sns.regplot(x=x, y=y, data=df, ax=ax)
            st.pyplot(fig)

    elif analysis_type == "Check Normality":
        st.subheader("Normality Test (Shapiro-Wilk)")
        col = st.sidebar.selectbox("Choose numeric column", numeric_cols)
        if col:
            stat, p = stats.shapiro(df[col].dropna())
            st.write(f"**W-statistic**: {stat:.3f}, **p-value**: {p:.3f}")
            st.write("Interpretation: If p > 0.05, the data appears normally distributed.")
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)

    elif analysis_type == "Compare Variances":
        st.subheader("Compare Variances (Levene's Test)")
        target = st.sidebar.selectbox("Choose numeric variable", numeric_cols)
        group = st.sidebar.selectbox("Choose grouping variable", categorical_cols)
        if target and group:
            unique_groups = df[group].dropna().unique()
            if len(unique_groups) == 2:
                g1 = df[df[group] == unique_groups[0]][target].dropna()
                g2 = df[df[group] == unique_groups[1]][target].dropna()
                stat, p = stats.levene(g1, g2)
                st.write(f"**Levene's statistic**: {stat:.3f}, **p-value**: {p:.3f}")
                st.write("Interpretation: If p > 0.05, variances are equal (assumption for t-test holds).")
                fig, ax = plt.subplots()
                sns.boxplot(x=group, y=target, data=df, ax=ax)
                st.pyplot(fig)

else:
    st.info("Upload a CSV file to begin.")
