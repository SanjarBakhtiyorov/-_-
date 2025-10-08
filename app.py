# -*- coding: utf-8 -*-
import io
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional

import streamlit as st
import pandas as pd
import numpy as np

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression

def format_numbers_ru(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format all numeric columns using Russian style: '# ##0,00'
    Example: 1234567.89 -> '1 234 567,89'
    """
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].apply(
                lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
                if pd.notnull(x) else ""
            )
    return df_fmt

st.set_page_config(page_title="Финансовый пайплайн | Artel Support", layout="wide")

# ==================== UI HEADER ====================
st.title("Финансовый пайплайн (G1/G2/G3, D-блок, UZS→USD, ML-скидки)")
st.caption("Загрузка Excel из CRM → нормализация → корректировки → своды → готовый Excel-отчёт")

# ==================== CONFIG ====================

@dataclass
class PipelineConfig:
    src_sheet: str = "Доставленный"
    excluded_for_manufacturer: Tuple[str, ...] = ("ВЫЗОВ", "ПРОДАЖА")
    spare_parts_labels: Tuple[str, ...] = ("Запчасть", )
    ml_threshold: float = 0.85
    rename_map: Dict[str, str] = field(default_factory=lambda: {
        "Номер заявки": "ticket_id",
        "Тип гарантии": "warranty_type",
        "Группа услуг": "service_group",
        "Группа услуг ": "service_group",
        "Услуга": "service_name",
        "Сумма по продукту": "sum_product",
        "Сумма по услуге": "sum_service",
        "Сумма по продукту Без скидки": "sum_product_before_disc",
        "Сумма по услуге Без скидки": "sum_service_before_disc",
        "Скидка на услугу": "discount_service",
        "Скидка на продукт": "discount_product",
        "завод": "plant_name",
        "Вид материала": "material_type",
        "Вид материала ": "material_type",
        "Бренд": "brand",
        "Дата создания": "created_at",
        "Дата продажи": "sold_at",
        "Сервисный центр": "service_center",
        "Сервисный центр ": "service_center",
        "Код продукта": "product_code",
        "Продукт": "product_name",
        "Тип заявки": "request_type",
        "Статус заявки": "ticket_status",
        "Статус заявки ": "ticket_status",
        "Техническое заключение": "tech_conclusion",
        "СПС 1": "sps1",
        "СПС 2": "sps2",
        "дата спс": "sps_date",
        "Код услуги": "service_code",
        "Стоимость услуги": "service_unit_price",
        "Количество услуг": "qty_service",
        "Дата оказания": "service_date",
        "Дата проведения": "post_date",
    })
    required_cols: Tuple[str, ...] = (
        "ticket_id","warranty_type","service_group","sum_product","sum_service",
        "discount_service","discount_product","plant_name","material_type",
        "service_code","service_unit_price","qty_service","sps_date"
    )

MONEY_FORMAT_RU = '# ##0,00'  # 1 234 567,89

# ==================== SIDEBAR ====================
st.sidebar.header("Настройки входных данных")
sheet_name = st.sidebar.text_input("Имя листа в Excel", value="Доставленный")
excluded_types = st.sidebar.text_input("Исключить из затрат завода (через запятую)", value="ВЫЗОВ, ПРОДАЖА")
sp_labels = st.sidebar.text_input("Метки запчастей (через запятую)", value="Запчасть")

st.sidebar.markdown("---")
st.sidebar.subheader("Курсы валют и ML скидок")
rates_file = st.sidebar.file_uploader("Курс UZS→USD (CSV/XLSX)", type=["csv", "xlsx"])
ml_training_file = st.sidebar.file_uploader("История скидок с метками (CSV/XLSX)", type=["csv", "xlsx"])
ml_threshold = st.sidebar.slider("Порог уверенности ML (скидки):", 0.5, 0.99, 0.85, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Загрузите CRM выгрузку")
uploaded = st.sidebar.file_uploader("Excel из CRM (.xlsx)", type=["xlsx"])

# ==================== CACHING HELPERS ====================
@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes, sheet: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)

@st.cache_data(show_spinner=False)
def read_rates(file) -> pd.DataFrame:
    # CSV/XLSX, find date/rate columns
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in ("date", "дата")), None)
    rate_col = next((cols[k] for k in cols if k in ("rate", "курс", "uzs_usd", "uzs_to_usd")), None)
    if not date_col or not rate_col:
        raise ValueError("В файле курсов должны быть колонки 'date/Дата' и 'rate/Курс'.")
    df = df.rename(columns={date_col: "date", rate_col: "rate"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["date", "rate"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df[["date","rate"]]

@st.cache_data(show_spinner=False)
def read_training_discounts(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    low = {c.strip().lower(): c for c in df.columns}
    desc_col = next((low[k] for k in low if k in ("description", "описание", "discount_desc", "desc")), None)
    label_col = next((low[k] for k in low if k in ("approved_by", "утверждено", "label", "метка")), None)
    if not desc_col or not label_col:
        raise ValueError("В обучающем датасете нужны колонки 'description/описание' и 'approved_by/утверждено'.")
    df = df.rename(columns={desc_col: "description", label_col: "approved_by"})
    df["description"] = df["description"].astype(str).fillna("")
    df["approved_by"] = df["approved_by"].astype(str).str.upper().str.strip()
    df = df.dropna(subset=["description","approved_by"])
    return df[["description","approved_by"]]

@st.cache_resource(show_spinner=False)
def train_discount_classifier(train_df: pd.DataFrame):
    pipe = SkPipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto"))
    ])
    pipe.fit(train_df["description"].values, train_df["approved_by"].values)
    return pipe

# ==================== PIPELINE FUNCTIONS ====================
def load_and_normalize(file_bytes: bytes, cfg: PipelineConfig, sheet_name: Optional[str] = None) -> pd.DataFrame:
    df = read_excel_cached(file_bytes, sheet_name or cfg.src_sheet)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in cfg.rename_map.items() if k in df.columns})

    for col in cfg.required_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["sum_product","sum_service","discount_service","discount_product","service_unit_price","qty_service"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["qty_service"] = df["qty_service"].fillna(1)
    df[["sum_product","sum_service","discount_service","discount_product","service_unit_price"]] = \
        df[["sum_product","sum_service","discount_service","discount_product","service_unit_price"]].fillna(0)

    # normalize texts/dates
    for dcol in ["sps_date","service_date","post_date","sold_at","created_at"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    df["service_group_up"] = df["service_group"].astype(str).str.upper().str.strip()
    df["warranty_type"] = df["warranty_type"].astype(str).str.upper().str.strip()
    df["material_type"] = df["material_type"].astype(str).str.strip()

    df["sum_total"] = df["sum_product"] + df["sum_service"]
    df["discount_total"] = df["discount_service"] + df["discount_product"]
    return df

def add_line_flags(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    df["is_excluded_for_manufacturer"] = df["service_group_up"].isin([s.upper() for s in cfg.excluded_for_manufacturer])
    df["is_spare_part"] = df["material_type"].isin(cfg.spare_parts_labels)
    return df

def apply_latest_service_price(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    elig = (df["warranty_type"].isin(["G1","G2"])) & (~df["is_spare_part"]) & df["service_code"].notna()
    df_elig = df.loc[elig].copy().sort_values(["service_code","sps_date"])

    latest_price_map = (
        df_elig.dropna(subset=["service_unit_price","sps_date"])
              .groupby("service_code")["service_unit_price"]
              .last().to_dict()
    )

    old_price = df.loc[elig, "service_unit_price"].copy()
    new_price = df.loc[elig, "service_code"].map(latest_price_map).fillna(old_price)
    qty = df.loc[elig, "qty_service"].fillna(1)

    old_sum_service = df.loc[elig, "sum_service"].fillna(0).astype(float)
    old_sum_service_recalc = (old_price.fillna(0).astype(float) * qty.astype(float))
    old_sum_service_final = np.where(old_sum_service > 0, old_sum_service, old_sum_service_recalc)
    new_sum_service = (new_price.astype(float) * qty.astype(float))
    delta = new_sum_service - old_sum_service_final

    df.loc[elig, "service_unit_price"] = new_price
    df.loc[elig, "sum_service"] = new_sum_service
    df.loc[elig, "sum_total"] = df.loc[elig, "sum_product"] + df.loc[elig, "sum_service"]

    audit = df.loc[elig, ["ticket_id","service_code","plant_name","warranty_type","sps_date","qty_service"]].copy()
    audit["old_unit_price"] = old_price.values
    audit["new_unit_price"] = new_price.values
    audit["old_sum_service"] = old_sum_service_final
    audit["new_sum_service"] = new_sum_service
    audit["delta_sum_service"] = delta
    audit = audit[audit["delta_sum_service"] != 0]  # comment out to see all
    return df, audit

def apply_usd_conversion(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    working_date = df["sps_date"]
    if working_date.isna().all():
        for fallback in ["service_date","post_date","sold_at","created_at"]:
            if fallback in df.columns:
                wd = pd.to_datetime(df[fallback], errors="coerce")
                if wd.notna().any():
                    working_date = wd
                    break
    df["_rate_date"] = pd.to_datetime(working_date, errors="coerce")

    tmp = df[["_rate_date"]].copy()
    tmp["__row__"] = np.arange(len(tmp))
    tmp = tmp.sort_values("_rate_date")
    rates_sorted = rates.sort_values("date")

    merged = pd.merge_asof(tmp, rates_sorted, left_on="_rate_date", right_on="date", direction="backward")
    merged = merged.sort_values("__row__")
    df["usd_rate"] = merged["rate"].values

    for col in ["sum_product","sum_service","sum_total","discount_product","discount_service","discount_total"]:
        if col in df.columns:
            df[col + "_usd"] = np.where(df["usd_rate"] > 0, df[col] / df["usd_rate"], np.nan)
    return df

def aggregate_by_ticket(df: pd.DataFrame, use_usd: bool=False) -> pd.DataFrame:
    if use_usd:
        return (
            df.groupby("ticket_id").agg(
                sum_product=("sum_product_usd","sum"),
                sum_service=("sum_service_usd","sum"),
                sum_total=("sum_total_usd","sum"),
                discount_product=("discount_product_usd","sum"),
                discount_service=("discount_service_usd","sum"),
                discount_total=("discount_total_usd","sum"),
                warranty_type=("warranty_type","first"),
                plant_name=("plant_name","first")
            ).reset_index()
        )
    else:
        return (
            df.groupby("ticket_id").agg(
                sum_product=("sum_product","sum"),
                sum_service=("sum_service","sum"),
                sum_total=("sum_total","sum"),
                discount_product=("discount_product","sum"),
                discount_service=("discount_service","sum"),
                discount_total=("discount_total","sum"),
                warranty_type=("warranty_type","first"),
                plant_name=("plant_name","first")
            ).reset_index()
        )

def warranty_totals_from_id(agg_id: pd.DataFrame) -> pd.DataFrame:
    return (
        agg_id.groupby("warranty_type")[["sum_product","sum_service","sum_total","discount_total"]]
        .sum().reset_index()
    )

def ar_by_plant(df: pd.DataFrame, use_usd: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["warranty_type"].isin(["G1","G2"]) & (~df["is_excluded_for_manufacturer"])
    cols = ["plant_name","warranty_type","ticket_id"]
    if use_usd:
        cols_vals = ["sum_product_usd","sum_service_usd","sum_total_usd","discount_total_usd"]
    else:
        cols_vals = ["sum_product","sum_service","sum_total","discount_total"]
    ar_lines = df.loc[mask, cols + cols_vals]
    ar_id = (
        ar_lines.groupby(["plant_name","warranty_type","ticket_id"], as_index=False)
        .agg(**{cols_vals[0]: (cols_vals[0], "sum"),
                cols_vals[1]: (cols_vals[1], "sum"),
                cols_vals[2]: (cols_vals[2], "sum"),
                cols_vals[3]: (cols_vals[3], "sum")})
    )
    ar_summary = (
        ar_id.groupby(["plant_name","warranty_type"], as_index=False)
        .agg(**{cols_vals[0]: (cols_vals[0], "sum"),
                cols_vals[1]: (cols_vals[1], "sum"),
                cols_vals[2]: (cols_vals[2], "sum"),
                cols_vals[3]: (cols_vals[3], "sum")})
        .sort_values(cols_vals[2], ascending=False)
    )
    return ar_id, ar_summary

def g3_views(df: pd.DataFrame, use_usd: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols_base = ["service_group","ticket_id","is_spare_part","plant_name"]
    if use_usd:
        vals = ["sum_product_usd","sum_service_usd","sum_total_usd","discount_total_usd"]
    else:
        vals = ["sum_product","sum_service","sum_total","discount_total"]
    g3_lines = df.loc[df["warranty_type"]=="G3", cols_base + vals].copy()
    g3_summary = (
        g3_lines.groupby("service_group", as_index=False)
        .agg(**{vals[0]: (vals[0],"sum"),
                vals[1]: (vals[1],"sum"),
                vals[2]: (vals[2],"sum"),
                vals[3]: (vals[3],"sum")})
        .sort_values(vals[2], ascending=False)
    )
    return g3_lines, g3_summary

def dblock_outputs(df: pd.DataFrame, use_usd: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask_pay = df["is_spare_part"] & df["warranty_type"].isin(["G2","G3"])
    if use_usd:
        pay_lines = df.loc[mask_pay, ["plant_name","warranty_type","ticket_id","product_name","product_code",
                                      "sum_product_usd","discount_product_usd"]].copy()
        pay_lines = pay_lines.rename(columns={"sum_product_usd":"sum_product","discount_product_usd":"discount_product"})
    else:
        pay_lines = df.loc[mask_pay, ["plant_name","warranty_type","ticket_id","product_name","product_code",
                                      "sum_product","discount_product"]].copy()
    pay_summary = (
        pay_lines.groupby(["plant_name","warranty_type"], as_index=False)
        .agg(Сумма_запчастей=("sum_product","sum"),
             Скидки_по_запчастям_инфо=("discount_product","sum"))
        .sort_values("Сумма_запчастей", ascending=False)
    )
    mask_disc = df["is_spare_part"] & ((df.get("discount_product",0)>0) | (df.get("discount_service",0)>0))
    if use_usd:
        disc_register = df.loc[mask_disc,
            ["plant_name","warranty_type","ticket_id","product_name","product_code",
             "discount_product_usd","discount_service_usd","sum_product_usd","sum_service_usd","sum_total_usd"]
        ].copy().rename(columns={
            "discount_product_usd":"discount_product","discount_service_usd":"discount_service",
            "sum_product_usd":"sum_product","sum_service_usd":"sum_service","sum_total_usd":"sum_total"
        })
    else:
        disc_register = df.loc[mask_disc,
            ["plant_name","warranty_type","ticket_id","product_name","product_code",
             "discount_product","discount_service","sum_product","sum_service","sum_total"]
        ].copy()
    return pay_lines, pay_summary, disc_register

def build_changed_prices_reports(df_before: pd.DataFrame, audit_price_adj: pd.DataFrame, cfg: PipelineConfig):
    changed_lines = audit_price_adj.rename(columns={
        "ticket_id":"Номер заявки","service_code":"Код услуги","plant_name":"Завод",
        "warranty_type":"Тип гарантии","sps_date":"Дата СПС","qty_service":"Кол-во услуг",
        "old_unit_price":"Старая цена услуги","new_unit_price":"Новая цена услуги",
        "old_sum_service":"Старая сумма по услуге","new_sum_service":"Новая сумма по услуге",
        "delta_sum_service":"Дельта (сумма по услуге)"
    })
    cols_order = [
        "Номер заявки","Код услуги","Тип гарантии","Завод","Дата СПС","Кол-во услуг",
        "Старая цена услуги","Новая цена услуги",
        "Старая сумма по услуге","Новая сумма по услуге","Дельта (сумма по услуге)"
    ]
    if not changed_lines.empty:
        changed_lines = changed_lines[[c for c in cols_order if c in changed_lines.columns]] \
            .sort_values(["Код услуги","Дата СПС","Номер заявки"], ascending=[True, True, True])

    elig_before = (
        (df_before["warranty_type"].isin(["G1","G2"])) &
        (~df_before["material_type"].isin(cfg.spare_parts_labels)) &
        (df_before["service_code"].notna())
    )
    last_sps_date = (
        df_before.loc[elig_before, ["service_code","sps_date"]]
        .dropna(subset=["sps_date"])
        .groupby("service_code", as_index=False)["sps_date"].max()
        .rename(columns={"sps_date": "Последняя дата СПС"})
    )

    def _mode_or_edge(s: pd.Series, edge: str = "last"):
        s = pd.Series(s)
        m = s.mode()
        if len(m): return m.iloc[0]
        return s.iloc[-1] if edge == "last" else s.iloc[0]

    if audit_price_adj.empty:
        changed_summary = pd.DataFrame(columns=["Код услуги","Старая_цена","Новая_цена",
                                                "Последняя дата СПС","Строк_затронуто","Уникальных_заявок","Дельта_сумма"])
        return changed_summary, changed_lines

    pairs = (
        audit_price_adj.groupby("service_code", as_index=False)
        .agg(
            Старая_цена=("old_unit_price", lambda x: _mode_or_edge(x, "first")),
            Новая_цена=("new_unit_price", lambda x: _mode_or_edge(x, "last")),
            Строк_затронуто=("service_code","count"),
            Уникальных_заявок=("ticket_id", pd.Series.nunique),
            Дельта_сумма=("delta_sum_service","sum"),
        )
        .rename(columns={"service_code":"Код услуги"})
    )
    changed_summary = pairs.merge(
        last_sps_date.rename(columns={"service_code":"Код услуги"}),
        on="Код услуги", how="left"
    )[["Код услуги","Старая_цена","Новая_цена","Последняя дата СПС","Строк_затронуто","Уникальных_заявок","Дельта_сумма"]] \
     .sort_values(["Последняя дата СПС","Код услуги"], ascending=[False, True])
    return changed_summary, changed_lines

def ensure_ticket_id_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ticket_id" in out.columns:
        def _to_text(x):
            if pd.isna(x):
                return ""
            try:
                as_num = pd.to_numeric(x, errors="coerce")
                if pd.isna(as_num):
                    return str(x)
                return str(int(as_num))
            except Exception:
                return str(x)
        out["ticket_id"] = out["ticket_id"].apply(_to_text)
    if "Номер заявки" in out.columns:
        out["Номер заявки"] = out["Номер заявки"].astype(str)
    return out

def write_sheets_to_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_money = wb.add_format({'num_format': MONEY_FORMAT_RU})
        fmt_text  = wb.add_format({'num_format': '@'})

        for sheet_name, df in sheets.items():
            df_to_write = ensure_ticket_id_text(df)
            df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]

            # Which columns look like money?
            money_cols = []
            for c in df_to_write.columns:
                lc = str(c).lower()
                if any(k in lc for k in ["sum", "сумм", "скидк", "цена", "дельта", "debt", "receiv", "payable", "usd"]):
                    money_cols.append(c)

            # Apply formats
            for i, name in enumerate(df_to_write.columns):
                if name in ("ticket_id", "Номер заявки"):
                    ws.set_column(i, i, 18, fmt_text)
                elif name in money_cols:
                    ws.set_column(i, i, 16, fmt_money)

            # Autofit without losing formats
            for i, name in enumerate(df_to_write.columns):
                vals = df_to_write.iloc[:, i].astype(str).head(200).tolist()
                width = min(max(10, max(len(str(name)), *(len(v) for v in vals)) + 2), 42)
                keep_fmt = fmt_text if name in ("ticket_id","Номер заявки") else (fmt_money if name in money_cols else None)
                ws.set_column(i, i, width, keep_fmt)

    output.seek(0)
    return output.getvalue()

# ==================== UI FILTERS BLOCK ====================
def apply_ui_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        plants = ["(все)"] + sorted([p for p in df["plant_name"].dropna().astype(str).unique()])
        plant_sel = st.selectbox("Фильтр: завод", plants)
        if plant_sel != "(все)":
            df = df[df["plant_name"] == plant_sel]
    with c2:
        brands = ["(все)"] + sorted([b for b in df.get("brand", pd.Series(dtype=str)).dropna().astype(str).unique()])
        brand_sel = st.selectbox("Фильтр: бренд", brands)
        if brand_sel != "(все)" and "brand" in df.columns:
            df = df[df["brand"] == brand_sel]
    with c3:
        warranties = ["(все)", "G1", "G2", "G3"]
        w_sel = st.selectbox("Фильтр: гарантия", warranties)
        if w_sel != "(все)":
            df = df[df["warranty_type"] == w_sel]
    with c4:
        # date filter on sps_date
        min_d = pd.to_datetime(df["sps_date"]).min()
        max_d = pd.to_datetime(df["sps_date"]).max()
        if pd.notna(min_d) and pd.notna(max_d):
            start, end = st.date_input("Период по 'дата спс'", value=(min_d.date(), max_d.date()))
            if start and end:
                mask = (pd.to_datetime(df["sps_date"]).dt.date >= start) & (pd.to_datetime(df["sps_date"]).dt.date <= end)
                df = df[mask]
    return df

# ==================== MAIN FLOW ====================

cfg = PipelineConfig(
    src_sheet=sheet_name,
    excluded_for_manufacturer=tuple([s.strip().upper() for s in excluded_types.split(",") if s.strip()]),
    spare_parts_labels=tuple([s.strip() for s in sp_labels.split(",") if s.strip()]),
    ml_threshold=ml_threshold
)

if uploaded is None:
    st.info("Загрузите Excel-файл CRM (формат .xlsx) в сайдбаре, чтобы начать.")
    st.stop()

try:
    with st.spinner("Читаем и нормализуем данные..."):
        df = load_and_normalize(uploaded.getvalue(), cfg, sheet_name=sheet_name)
        df = add_line_flags(df, cfg)
        df_before = df.copy()

    # Optional UI filters (preview layer). We’ll apply them only to previews (not to final export).
    st.markdown("### Фильтры предпросмотра")
    df_preview = apply_ui_filters(df)

    with st.spinner("Корректируем цены G1/G2 по последним 'дата спс'..."):
        df, audit_price_adj = apply_latest_service_price(df, cfg)
        changed_summary, changed_lines = build_changed_prices_reports(df_before, audit_price_adj, cfg)

    # Currency conversion (optional)
    usd_enabled = False
    if rates_file is not None:
        with st.spinner("Применяем курсы UZS→USD..."):
            rates = read_rates(rates_file)
            df = apply_usd_conversion(df, rates)
            usd_enabled = True
    else:
        st.info("Курсы валют не загружены — USD-агрегации будут пропущены.")

    # Aggregations (UZS)
    with st.spinner("Строим своды (UZS)..."):
        agg_id = aggregate_by_ticket(df, use_usd=False)
        warranty_totals = warranty_totals_from_id(agg_id)
        ar_id, ar_summary = ar_by_plant(df, use_usd=False)
        g3_lines, g3_summary = g3_views(df, use_usd=False)
        dblock_pay_lines, dblock_pay_summary, dblock_disc_register = dblock_outputs(df, use_usd=False)

    # Aggregations (USD), if enabled
    if usd_enabled:
        with st.spinner("Строим своды (USD)..."):
            agg_id_usd = aggregate_by_ticket(df, use_usd=True)
            warranty_totals_usd = warranty_totals_from_id(agg_id_usd)
            ar_id_usd, ar_summary_usd = ar_by_plant(df, use_usd=True)
            g3_lines_usd, g3_summary_usd = g3_views(df, use_usd=True)
            dblock_pay_lines_usd, dblock_pay_summary_usd, dblock_disc_register_usd = dblock_outputs(df, use_usd=True)
    else:
        agg_id_usd = warranty_totals_usd = ar_id_usd = ar_summary_usd = None
        g3_lines_usd = g3_summary_usd = dblock_pay_lines_usd = dblock_pay_summary_usd = dblock_disc_register_usd = None

    # ML classification (optional)
    confident_ml, review_ml = pd.DataFrame(), pd.DataFrame()
    if ml_training_file is not None:
        with st.spinner("Обучаем модель скидок и применяем к текущим данным..."):
            train_df = read_training_discounts(ml_training_file)
            model = train_discount_classifier(train_df)
            # apply on original df (post-price-adjustment)
            confident_ml, review_ml = (lambda d: (pd.DataFrame(), pd.DataFrame()))(df)
            # Try classify on df with discount columns present:
            # We’ll prefer description from likely columns
            def classify_discounts(current_df: pd.DataFrame, model, threshold: float = 0.85):
                dfc = current_df.copy()
                text_col = None
                for candidate in ["discount_description","описание скидки","описание","description","desc","sps1","sps2","service_name"]:
                    if candidate in dfc.columns:
                        text_col = candidate
                        break
                if text_col is None:
                    return pd.DataFrame(), pd.DataFrame()
                has_disc = ((dfc.get("discount_product", 0) > 0) | (dfc.get("discount_service", 0) > 0))
                cand = dfc.loc[has_disc].copy()
                if cand.empty:
                    return pd.DataFrame(), pd.DataFrame()
                texts = cand[text_col].astype(str).fillna("")
                try:
                    proba = model.predict_proba(texts)
                    classes = model.classes_
                    pred_idx = np.argmax(proba, axis=1)
                    cand["ML_Метка"] = classes[pred_idx]
                    cand["ML_Уверенность"] = proba[np.arange(len(cand)), pred_idx]
                except Exception:
                    pred = model.predict(texts)
                    cand["ML_Метка"] = pred
                    cand["ML_Уверенность"] = np.nan
                confident_df = cand.loc[cand["ML_Уверенность"] >= ml_threshold].copy()
                review_df = cand.loc[(cand["ML_Уверенность"] < ml_threshold) | (cand["ML_Уверенность"].isna())].copy()
                return confident_df, review_df

            confident_ml, review_ml = classify_discounts(df, model, threshold=ml_threshold)
    else:
        st.info("Не загружен обучающий датасет скидок — классификация пропущена.")

    # ==================== PREVIEW ====================
    st.success("Готово! Предпросмотр ключевых сводок ниже, а также кнопка скачивания Excel.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Итоги по гарантии (UZS)")
        st.dataframe(format_numbers_ru(warranty_totals.head(200))
        st.subheader("AR по заводам — свод (UZS)")
        st.dataframe(format_numbers_ru(ar_summary.head(200))
        if usd_enabled:
            st.subheader("Итоги по гарантии (USD)")
            st.dataframe(format_numbers_ru(warranty_totals_usd.head(200))
            st.subheader("AR по заводам — свод (USD)")
            st.dataframe(format_numbers_ru(ar_summary_usd.head(200))
    with c2:
        st.subheader("G3 — свод (UZS)")
        st.dataframe(format_numbers_ru(g3_summary.head(200))
        if usd_enabled:
            st.subheader("G3 — свод (USD)")
            st.dataframe(format_numbers_ru(g3_summary_usd.head(200))
        st.subheader("Изменённые цены (свод)")
        st.dataframe(format_numbers_ru(changed_summary.head(200))
        if not confident_ml.empty:
            st.subheader("Скидки (ML, уверенные предсказания)")
            st.dataframe(format_numbers_ru(confident_ml.head(200))
        if not review_ml.empty:
            st.subheader("Скидки (на ручную проверку)")
            st.dataframe(format_numbers_ru(review_ml.head(200))

    # ==================== EXCEL EXPORT ====================
    sheets = {
        "Итоги_по_ID": agg_id,
        "Итоги_по_Гарантии": warranty_totals,
        "АР_по_Заводам_ID": ar_id,
        "АР_по_Заводам_Свод": ar_summary,
        "Г3_Строки": g3_lines,
        "Г3_Свод": g3_summary,
        "ДБлок_Кредиторка_Строки": dblock_pay_lines,
        "ДБлок_Кредиторка_Свод": dblock_pay_summary,
        "ДБлок_Скидки": dblock_disc_register,
        "G1G2_Корректировка_Тарифов": audit_price_adj,
        "G1G2_Измененные_Цены_Свод": changed_summary,
        "G1G2_Измененные_Цены_Строки": changed_lines,
    }

    if usd_enabled:
        sheets.update({
            "Итоги_по_ID_USD": agg_id_usd,
            "Итоги_по_Гарантии_USD": warranty_totals_usd,
            "АР_по_Заводам_ID_USD": ar_id_usd,
            "АР_по_Заводам_Свод_USD": ar_summary_usd,
            "Г3_Строки_USD": g3_lines_usd,
            "Г3_Свод_USD": g3_summary_usd,
            "ДБлок_Кредиторка_Строки_USD": dblock_pay_lines_usd,
            "ДБлок_Кредиторка_Свод_USD": dblock_pay_summary_usd,
            "ДБлок_Скидки_USD": dblock_disc_register_usd,
        })

    if not confident_ml.empty:
        sheets["Скидки_Классификация"] = confident_ml
    if not review_ml.empty:
        sheets["Скидки_К_Проверке"] = review_ml

    xlsx_bytes = write_sheets_to_bytes(sheets)
    st.download_button(
        label="💾 Скачать готовый Excel-отчёт",
        data=xlsx_bytes,
        file_name="Финансовые_итоги_месяца.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"Ошибка: {e}")
