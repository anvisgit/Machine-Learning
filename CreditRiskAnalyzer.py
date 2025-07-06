# to add 1.smote, imbalenced data shi 2.shap 3. more graphs 4. shap library 5.business pov

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="🪙", layout="wide")

st.title("Credit Risk Analyser")
uploaded = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
st.sidebar.title("Settings")

st.sidebar.markdown("___")


st.sidebar.subheader("Pick a Model:")
pick = st.sidebar.radio("(any one) ", ["SVM", "Random Forest", "Logistic Regression"])
st.sidebar.subheader("Select atleast One:")
show_heatmap = st.sidebar.checkbox("Show Confusion Matrix")
show_roc = st.sidebar.checkbox("Show ROC Curve")
show_eval = st.sidebar.checkbox("Show Evaluation Report")
st.sidebar.subheader("\n")
st.sidebar.markdown("___")

st.sidebar.markdown("Made by:")
st.sidebar.markdown(
    """
<a href=" https://www.linkedin.com/in/anvi-wadhwa-10a1a1329?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
    <button style='font-size:16px;padding:8px 16px;background-color:#0077B5;color:white;border:none;border-radius:5px;'>
        Anvi Wadhwa
    </button>
</a>
""",
    unsafe_allow_html=True,
)
st.sidebar.subheader("\n")

if uploaded:
    deets = {
        "filename": uploaded.name,
        "filesize": uploaded.size,
        "filetype": uploaded.type,
    }
    st.subheader("File Details:")
    st.write(deets)

    if uploaded.type == "text/csv":
        df = pd.read_csv(uploaded)
        st.write("Preview of Data")
        st.write(df.head())
        df = df.dropna()
    else:
        st.write("Input File must be csv or text")

    col1, col2 = st.columns(2)

    with col1:
        fig1, axis1 = plt.subplots(figsize=(4, 3))
        sns.lineplot(x="age", y="income", data=df, ax=axis1)
        axis1.set_title("Age vs Income")
        axis1.set_xlabel("Age")
        axis1.set_ylabel("Income")
        st.pyplot(fig1)

    with col2:
        fig2, axis2 = plt.subplots(figsize=(4, 3))
        sns.lineplot(x="age", y="debtinc", data=df, ax=axis2)
        axis2.set_title("Age vs Debt-Income")
        axis2.set_xlabel("Age")
        axis2.set_ylabel("Debt-Income")
        st.pyplot(fig2)

    x = df.drop(["default"], axis=1)
    y = df["default"]

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, random_state=2
    )

    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)

    if pick == "Random Forest":
        rfc = RandomForestClassifier(n_estimators=300)
        rfc.fit(train_x, train_y)
        netscore = rfc.score(test_x, test_y)
        st.subheader(f"NetScore (Random Forest Test Accuracy): {netscore:.4f}")
        yp = rfc.predict(test_x)
        prob_default = rfc.predict_proba(test_x)[:, 1]

        results_df = pd.DataFrame(sc.inverse_transform(test_x), columns=x.columns)
        results_df["Predicted Default"] = yp
        results_df["Probability of Default"] = prob_default
        results_df["Actual Default"] = test_y.values

        st.subheader(" Individual Credit Risk Predictions (Random Forest)")
        st.dataframe(
            results_df.sort_index(ascending=True)
        )

        if show_heatmap:
            c = confusion_matrix(test_y, yp)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(c, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Random Forest Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        if show_roc:

            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
            if hasattr(rfc, "predict_proba"):
                y_prob_rf = rfc.predict_proba(test_x)[:, 1]
                fpr_rf, tpr_rf, _ = roc_curve(test_y, y_prob_rf)
                auc_rf = auc(fpr_rf, tpr_rf)
                ax_roc.plot(
                    fpr_rf,
                    tpr_rf,
                    label=f"Random Forest (AUC = {auc_rf:.2f})",
                    linewidth=2,
                )

            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Combined ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        if show_eval:
            st.subheader("Evaluation Reports (Test Set)")
            yp_rf = rfc.predict(test_x)
            report_rf = classification_report(test_y, yp_rf, output_dict=True)
            df_rf = pd.DataFrame(report_rf).transpose()
            st.markdown("Random Forest")
            st.dataframe(df_rf.style.format("{:.2f}"))

    elif pick == "Logistic Regression":
        lr = LogisticRegression()
        lr.fit(train_x, train_y)
        netscore = lr.score(test_x, test_y)
        st.subheader(f"NetScore (Logistic Regression Accuracy): {netscore:.4f}")
        yp = lr.predict(test_x)
        prob_default = lr.predict_proba(test_x)[:, 1]

        results_df = pd.DataFrame(test_x, columns=x.columns)
        results_df["Predicted Default"] = yp
        results_df["Probability of Default"] = prob_default
        results_df["Actual Default"] = test_y.values

        st.subheader("Individual Credit Risk Predictions (Logistic Regression)")
        st.dataframe(
            results_df.sort_index( ascending=False)
        )

        if show_heatmap:
            c = confusion_matrix(test_y, yp)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(c, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Logistic Regression Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        if show_roc:
            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))

            if hasattr(lr, "predict_proba"):
                y_prob_lr = lr.predict_proba(test_x)[:, 1]
                fpr_lr, tpr_lr, _ = roc_curve(test_y, y_prob_lr)
                auc_lr = auc(fpr_lr, tpr_lr)
                ax_roc.plot(
                    fpr_lr,
                    tpr_lr,
                    label=f"Logistic Regression (AUC = {auc_lr:.2f})",
                    linewidth=2,
                )
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Combined ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        if show_eval:
            st.subheader("Evaluation Reports (Test Set)")
            yp_lr = lr.predict(test_x)
            report_lr = classification_report(test_y, yp_lr, output_dict=True)
            df_lr = pd.DataFrame(report_lr).transpose()
            st.markdown("Logistic Regression")
            st.dataframe(df_lr.style.format("{:.2f}"))

    elif pick == "SVM":
        sv = SVC(kernel="rbf", C=1.0, probability=True)
        sv.fit(train_x, train_y)
        sv_score = sv.score(test_x, test_y)
        st.subheader(f"NetScore (SVM Test Accuracy): {sv_score:.4f}")
        yp = sv.predict(test_x)
        try:
            prob_default = sv.predict_proba(test_x)[:, 1]
        except:
            prob_default = sv.decision_function(test_x)

        results_df = pd.DataFrame(test_x, columns=x.columns)
        results_df["Predicted Default"] = yp
        results_df["Probability of Default"] = prob_default
        results_df["Actual Default"] = test_y.values

        st.subheader("Individual Credit Risk Predictions (SVM)")
        st.dataframe(
            results_df.sort_index( ascending=True)
        )
        if show_heatmap:
            c = confusion_matrix(test_y, yp)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(c, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("SVM Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        if show_roc:
            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
            if hasattr(sv, "predict_proba"):
                y_prob_svm = sv.predict_proba(test_x)[:, 1]
            else:
                y_prob_svm = sv.decision_function(test_x)
            fpr_svm, tpr_svm, _ = roc_curve(test_y, y_prob_svm)
            auc_svm = auc(fpr_svm, tpr_svm)
            ax_roc.plot(
                fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.2f})", linewidth=2
            )

            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Combined ROC Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        if show_eval:
            st.subheader("Evaluation Reports (Test Set)")

            yp_svm = sv.predict(test_x)
            report_svm = classification_report(test_y, yp_svm, output_dict=True)
            df_svm = pd.DataFrame(report_svm).transpose()
            st.markdown("SVM")
            st.dataframe(df_svm.style.format("{:.2f}"))
