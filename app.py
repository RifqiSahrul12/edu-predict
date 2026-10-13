import io
import os
from pathlib import Path

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Prediksi Prestasi Akademik Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_FILE = "data.xlsx"
TARGET = "Kategori Prestasi"

# Kolom yang tidak boleh menjadi prediktor.
# "No" hanya penanda baris dan bersifat opsional.
NON_FEATURE_COLUMNS = {
    "No",
    "ID",
    "Nama",
    "Kelas",
    "Jenis Kelamin",
    TARGET,
    "Prediksi Prestasi",
}

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
<style>
.stApp { background:#f7f9fc; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }
header {
    visibility: visible;
    background: transparent !important;
}

section[data-testid="stSidebar"] {
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #173b68, #0f2948) !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}
.block-container {
    padding-top:1rem;
    max-width:1450px;
}

.title-box {
    background:#eaf3ff;
    border:1px solid #9fc5ee;
    border-radius:12px;
    text-align:center;
    padding:10px;
    font-size:22px;
    font-weight:700;
    margin-bottom:16px;
}

.hero, .card {
    background:white;
    border-radius:16px;
    border:1px solid #dfe5ec;
    padding:20px;
    margin-bottom:16px;
}

.metric-card {
    background:white;
    border:1px solid #dfe5ec;
    border-radius:14px;
    padding:16px;
    text-align:center;
}

.stButton > button {
    width:100%;
    border:none;
    border-radius:10px;
    background:#1f5f9f;
    color:white;
    font-weight:700;
    padding:11px;
}

.stButton > button:hover {
    background:#174a7d;
}

.small-note {
    color:#64748b;
    font-size:.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# BASIC HELPERS
# ============================================================
def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )
    return df


def normalize_target(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    aliases = {
        "tinggi": "Tinggi",
        "sedang": "Sedang",
        "rendah": "Rendah",
    }

    return aliases.get(text, str(value).strip().title())


def parse_attendance(value):
    """
    Normalisasi kehadiran ke 0..1.

    Contoh:
    96%  -> 0.96
    96   -> 0.96
    0.96 -> 0.96
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        text = value.strip().replace(",", ".")

        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100
            except ValueError:
                return np.nan

        try:
            number = float(text)
        except ValueError:
            return np.nan
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return np.nan

    if number > 1:
        number /= 100

    return number


def find_attendance_column(columns):
    for col in columns:
        text = str(col).lower()

        if "kehadiran" in text or "hadir" in text:
            return col

    return None


def get_feature_columns(df):
    return [
        col
        for col in df.columns
        if col not in NON_FEATURE_COLUMNS
    ]


# ============================================================
# FEATURE PREPROCESSING
# ============================================================
def fit_preprocessor(df, feature_columns):
    X = df[feature_columns].copy()

    attendance_col = find_attendance_column(X.columns)

    if attendance_col:
        X[attendance_col] = X[attendance_col].map(parse_attendance)

    encoders = {}
    medians = {}

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            values = X[col].astype("string")

            categories = sorted(
                values.dropna().unique().tolist()
            )

            mapping = {
                category: index
                for index, category in enumerate(categories)
            }

            X[col] = values.map(mapping).astype(float)
            encoders[col] = mapping

        X[col] = X[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        median = X[col].median()

        if pd.isna(median):
            median = 0.0

        medians[col] = float(median)
        X[col] = X[col].fillna(median)

    return X.astype(float), encoders, medians


def transform_features(
    df,
    feature_columns,
    encoders,
    medians,
):
    X = df[feature_columns].copy()

    attendance_col = find_attendance_column(X.columns)

    if attendance_col:
        X[attendance_col] = X[attendance_col].map(parse_attendance)

    for col in feature_columns:
        if col in encoders:
            values = X[col].astype("string")
            X[col] = values.map(encoders[col]).astype(float)
        else:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

        X[col] = X[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        X[col] = X[col].fillna(medians[col])

    return X.astype(float)


# ============================================================
# DATASET / MODEL
# ============================================================
def load_labeled_dataset():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"File '{DATASET_FILE}' tidak ditemukan."
        )

    df = pd.read_excel(DATASET_FILE)
    df = normalize_columns(df)

    if TARGET not in df.columns:
        raise ValueError(
            f"Kolom '{TARGET}' wajib tersedia pada data.xlsx."
        )

    df[TARGET] = df[TARGET].map(normalize_target)

    if df[TARGET].isna().any():
        raise ValueError(
            f"Kolom '{TARGET}' masih memiliki data kosong."
        )

    if df[TARGET].nunique() < 2:
        raise ValueError(
            "Dataset harus memiliki minimal dua kategori prestasi."
        )

    feature_columns = get_feature_columns(df)

    if not feature_columns:
        raise ValueError(
            "Tidak ditemukan kolom prediktor."
        )

    return df, feature_columns


def build_model():
    df, feature_columns = load_labeled_dataset()

    X, encoders, medians = fit_preprocessor(
        df,
        feature_columns,
    )

    y_text = df[TARGET].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    # Split otomatis. Tidak ada upload training/testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    evaluation_model = GaussianNB()
    evaluation_model.fit(
        X_train,
        y_train,
    )

    y_pred = evaluation_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    # Model final menggunakan seluruh data berlabel.
    final_model = GaussianNB()
    final_model.fit(
        X,
        y,
    )

    return {
        "model": final_model,
        "feature_columns": feature_columns,
        "encoders": encoders,
        "medians": medians,
        "label_encoder": label_encoder,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "y_test": y_test,
            "y_pred": y_pred,
            "labels": label_encoder.classes_,
            "train_size": len(X_train),
            "test_size": len(X_test),
        },
        "data": df,
    }


@st.cache_resource
def get_model_bundle(dataset_mtime):
    return build_model()


# ============================================================
# PREDICTION
# ============================================================
def predict_dataframe(df_input, bundle):
    df_input = normalize_columns(df_input)

    feature_columns = bundle["feature_columns"]

    missing = [
        col
        for col in feature_columns
        if col not in df_input.columns
    ]

    if missing:
        raise ValueError(
            "Kolom fitur belum lengkap: "
            + ", ".join(missing)
        )

    X = transform_features(
        df_input,
        feature_columns,
        bundle["encoders"],
        bundle["medians"],
    )

    model = bundle["model"]

    predicted_encoded = model.predict(X)

    predicted_labels = (
        bundle["label_encoder"]
        .inverse_transform(predicted_encoded)
    )

    probabilities = model.predict_proba(X)

    result = df_input.copy()

    # Jangan membawa target lama jika user mengunggah file checkpoint.
    if TARGET in result.columns:
        result = result.drop(columns=[TARGET])

    result["Prediksi Prestasi"] = predicted_labels

    for index, label in enumerate(
        bundle["label_encoder"].classes_
    ):
        result[
            f"Probabilitas {label} (%)"
        ] = probabilities[:, index] * 100

    return result


def dataframe_to_excel(df):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Hasil Prediksi",
        )

    return output.getvalue()


# ============================================================
# TEMPLATE
# ============================================================
def create_prediction_template(bundle):
    columns = []

    if "No" in bundle["data"].columns:
        columns.append("No")

    columns.append("Nama")
    columns.extend(bundle["feature_columns"])

    template = pd.DataFrame(
        columns=columns
    )

    return dataframe_to_excel(template)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
<h2 style='margin-bottom:4px;'>🎓 Prediksi Prestasi</h2>
<p style='font-size:14px;opacity:.9;'>Naïve Bayes</p>
""",
    unsafe_allow_html=True,
)

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Home",
        "Analisis Model",
        "Prediksi Prestasi",
        "Hasil Prediksi",
    ],
)

# ============================================================
# HOME
# ============================================================
if menu == "Home":

    st.markdown(
        "<div class='title-box'>"
        "PREDIKSI PRESTASI AKADEMIK SISWA"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='hero'>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([4, 1])

    with c1:
        st.title(
            "Prediksi Prestasi Akademik Siswa"
        )

        st.subheader(
            "Menggunakan Metode Naïve Bayes"
        )

        st.write(
            "Studi Kasus: MI Plus Andalusia Quranic School"
        )

        st.write(
            "Sistem mempelajari pola dari data siswa "
            "berlabel dan menghasilkan prediksi kategori "
            "prestasi pada data tanpa label."
        )

    with c2:
        st.markdown(
            "<div style='text-align:center;"
            "font-size:90px;margin-top:20px;'>🎓</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        bundle = get_model_bundle(
            os.path.getmtime(DATASET_FILE)
        )

        metrics = bundle["metrics"]

        cards = [
            (
                "Data Berlabel",
                len(bundle["data"]),
            ),
            (
                "Akurasi",
                f"{metrics['accuracy']:.2%}",
            ),
            (
                "Precision",
                f"{metrics['precision']:.2%}",
            ),
            (
                "F1-Score",
                f"{metrics['f1']:.2%}",
            ),
        ]

        cols = st.columns(4)

        for col, (label, value) in zip(
            cols,
            cards,
        ):
            with col:
                st.markdown(
                    f"""
<div class='metric-card'>
<div>{label}</div>
<h2>{value}</h2>
</div>
""",
                    unsafe_allow_html=True,
                )

    except Exception as error:
        st.warning(
            f"Model belum siap: {error}"
        )

    st.markdown(
        "<div class='card'>",
        unsafe_allow_html=True,
    )

    st.subheader("Alur Sistem")

    st.write(
        "Data berlabel → preprocessing → "
        "training/testing otomatis → model final → "
        "input/upload data tanpa kategori → "
        "prediksi → export hasil."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# ANALISIS MODEL
# ============================================================
elif menu == "Analisis Model":

    st.markdown(
        "<div class='title-box'>"
        "ANALISIS MODEL NAÏVE BAYES"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        bundle = get_model_bundle(
            os.path.getmtime(DATASET_FILE)
        )

        df = bundle["data"]
        metrics = bundle["metrics"]

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True,
        )

        st.subheader(
            "1. Dataset Berlabel"
        )

        st.write(
            "Dataset dibaca otomatis dari data.xlsx. "
            "Pembagian training dan testing dilakukan "
            "secara otomatis oleh sistem."
        )

        st.dataframe(
            df,
            use_container_width=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Jumlah Data",
            len(df),
        )

        c2.metric(
            "Training",
            metrics["train_size"],
        )

        c3.metric(
            "Testing",
            metrics["test_size"],
        )

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True,
        )

        st.subheader(
            "2. Distribusi Kategori Prestasi"
        )

        counts = df[TARGET].value_counts()

        fig, ax = plt.subplots(
            figsize=(7, 3.5)
        )

        counts.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_xlabel(
            "Kategori Prestasi"
        )

        ax.set_ylabel(
            "Jumlah Data"
        )

        ax.tick_params(
            axis="x",
            rotation=0,
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True,
        )

        st.subheader(
            "3. Variabel Prediktor"
        )

        st.write(
            "Kolom berikut digunakan sebagai prediktor:"
        )

        st.code(
            "\n".join(
                bundle["feature_columns"]
            )
        )

        st.info(
            "Kolom No/ID/Nama/Kelas/Jenis Kelamin "
            "tidak digunakan sebagai prediktor. "
            "Kategori Prestasi merupakan target."
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='card'>",
            unsafe_allow_html=True,
        )

        st.subheader(
            "4. Evaluasi Model"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy']:.2%}",
        )

        c2.metric(
            "Precision",
            f"{metrics['precision']:.2%}",
        )

        c3.metric(
            "Recall",
            f"{metrics['recall']:.2%}",
        )

        c4.metric(
            "F1-Score",
            f"{metrics['f1']:.2%}",
        )

        labels = list(metrics["labels"])

        cm = confusion_matrix(
            metrics["y_test"],
            metrics["y_pred"],
            labels=np.arange(len(labels)),
        )

        fig, ax = plt.subplots(
            figsize=(6, 4)
        )

        ax.imshow(cm)

        ax.set_xticks(
            range(len(labels))
        )

        ax.set_yticks(
            range(len(labels))
        )

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set_xlabel(
            "Prediksi"
        )

        ax.set_ylabel(
            "Aktual"
        )

        ax.set_title(
            "Confusion Matrix"
        )

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False,
        )

        plt.close(fig)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    except Exception as error:
        st.error(
            f"Analisis model gagal: {error}"
        )

# ============================================================
# PREDIKSI PRESTASI
# ============================================================
elif menu == "Prediksi Prestasi":

    st.markdown(
        "<div class='title-box'>"
        "PREDIKSI PRESTASI"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        bundle = get_model_bundle(
            os.path.getmtime(DATASET_FILE)
        )

        tab_upload, tab_form = st.tabs(
            [
                "Upload Excel",
                "Input Form",
            ]
        )

        # ----------------------------------------------------
        # UPLOAD EXCEL
        # ----------------------------------------------------
        with tab_upload:

            st.subheader(
                "Upload Data Tanpa Kategori"
            )

            st.write(
                "Upload file Excel yang berisi "
                "variabel prediktor. Kolom Kategori "
                "Prestasi tidak perlu diisi."
            )

            st.download_button(
                "Download Template Excel",
                data=create_prediction_template(
                    bundle
                ),
                file_name="template_prediksi.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

            uploaded = st.file_uploader(
                "Pilih file Excel",
                type=["xlsx", "xls"],
                key="prediction_upload",
            )

            if uploaded:

                df_input = normalize_columns(
                    pd.read_excel(uploaded)
                )

                missing = [
                    col
                    for col in bundle["feature_columns"]
                    if col not in df_input.columns
                ]

                if missing:

                    st.error(
                        "File belum sesuai template. "
                        "Kolom yang belum tersedia: "
                        + ", ".join(missing)
                    )

                else:

                    st.success(
                        f"{len(df_input)} data berhasil dibaca."
                    )

                    st.dataframe(
                        df_input,
                        use_container_width=True,
                    )

                    if st.button(
                        "🔍 Jalankan Prediksi",
                        key="run_upload_prediction",
                    ):

                        try:
                            result = predict_dataframe(
                                df_input,
                                bundle,
                            )

                            st.session_state[
                                "prediction_result"
                            ] = result

                            st.success(
                                "Prediksi berhasil dilakukan."
                            )

                            st.dataframe(
                                result,
                                use_container_width=True,
                            )

                            st.download_button(
                                "⬇️ Download Hasil Prediksi",
                                data=dataframe_to_excel(
                                    result
                                ),
                                file_name=(
                                    "hasil_prediksi.xlsx"
                                ),
                                mime=(
                                    "application/vnd.openxmlformats-officedocument."
                                    "spreadsheetml.sheet"
                                ),
                                key="download_upload_result",
                            )

                        except Exception as error:
                            st.error(
                                f"Prediksi gagal: {error}"
                            )

        # ----------------------------------------------------
        # INPUT FORM
        # ----------------------------------------------------
        with tab_form:

            st.subheader(
                "Input Data"
            )

            st.caption(
                "Form hanya menggunakan variabel prediktor. "
                "Biodata siswa tidak diperlukan."
            )

            form_values = {}

            form_cols = st.columns(2)

            for index, col in enumerate(
                bundle["feature_columns"]
            ):

                with form_cols[index % 2]:

                    attendance = (
                        find_attendance_column(
                            [col]
                        )
                        == col
                    )

                    if attendance:

                        form_values[col] = (
                            st.number_input(
                                f"{col} (%)",
                                min_value=0.0,
                                max_value=100.0,
                                value=0.0,
                                step=0.1,
                            )
                        )

                    else:

                        form_values[col] = (
                            st.number_input(
                                col,
                                value=0.0,
                                step=1.0,
                            )
                        )

            if st.button(
                "🔍 Prediksi",
                key="run_form_prediction",
            ):

                try:

                    one_row = pd.DataFrame(
                        [form_values]
                    )

                    result = predict_dataframe(
                        one_row,
                        bundle,
                    )

                    st.session_state[
                        "prediction_result"
                    ] = result

                    predicted = result.iloc[0][
                        "Prediksi Prestasi"
                    ]

                    st.markdown(
                        "<div class='card'>",
                        unsafe_allow_html=True,
                    )

                    st.subheader(
                        "Hasil Prediksi"
                    )

                    st.markdown(
                        f"<h1 style='text-align:center;'>"
                        f"{predicted}"
                        f"</h1>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    st.subheader(
                        "Probabilitas"
                    )

                    for label in (
                        bundle["label_encoder"].classes_
                    ):

                        column = (
                            f"Probabilitas {label} (%)"
                        )

                        probability = float(
                            result.iloc[0][column]
                        )

                        st.progress(
                            min(
                                1.0,
                                max(
                                    0.0,
                                    probability / 100,
                                ),
                            )
                        )

                        st.write(
                            f"{label}: "
                            f"{probability:.2f}%"
                        )

                    st.download_button(
                        "⬇️ Download Hasil",
                        data=dataframe_to_excel(
                            result
                        ),
                        file_name=(
                            "hasil_prediksi.xlsx"
                        ),
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        ),
                        key="download_form_result",
                    )

                except Exception as error:
                    st.error(
                        f"Prediksi gagal: {error}"
                    )

    except Exception as error:

        st.error(
            f"Prediksi belum dapat dijalankan: {error}"
        )

# ============================================================
# HASIL PREDIKSI
# ============================================================
elif menu == "Hasil Prediksi":

    st.markdown(
        "<div class='title-box'>"
        "HASIL PREDIKSI"
        "</div>",
        unsafe_allow_html=True,
    )

    result = st.session_state.get(
        "prediction_result"
    )

    if result is None or result.empty:

        st.info(
            "Belum ada hasil prediksi pada sesi ini."
        )

    else:

        st.dataframe(
            result,
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download Hasil Prediksi",
            data=dataframe_to_excel(
                result
            ),
            file_name="hasil_prediksi.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<hr>
<center>
<small>
Prediksi Prestasi Akademik Siswa •
MI Plus Andalusia Quranic School • 2026
</small>
</center>
""",
    unsafe_allow_html=True,
)
