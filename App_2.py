# -*- coding: utf-8 -*-
import io
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np

# ---------- Optional deps ----------
try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

SKLEARN_OK, SKLEARN_ERR = True, ""
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    SKLEARN_OK, SKLEARN_ERR = False, str(e)

# ---------- Page ----------
st.set_page_config(page_title="Финансовый пайплайн | Artel Support", layout="wide")
st.title("Финансовый пайплайн (G1/G2/G3, D-блок, UZS→USD, ML-скидки)")
st.caption("Загрузка Excel из CRM → нормализация → корректировки → своды → готовый Excel-отчёт")

MONEY_FORMAT_RU = '# ##0,00'  # Excel money format
EXCLUDE_NUM_FMT = {"ticket_id", "Номер заявки", "service_code", "Код услуги", "product_code", "Код продукта"}

# ---------- Utilities ----------
def _colnum_to_excel(n: int) -> str:
    s = ""; n += 1
    while n:
        n, r = divmod(n-1, 26)
        s = chr(65 + r) + s
    return s

def is_amount_col(colname: str) -> bool:
    lc = str(colname).lower()
    return any(k in lc for k in ["sum", "total", "скидк", "сумм", "цена", "дельта", "usd"])

def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    counts = out.columns.value_counts()
    for name in counts[counts > 1].index:
        cols = [c for c in out.columns if c == name]
        out[name] = out[cols].bfill(axis=1).iloc[:, 0]
        for c in cols[1:]:
            out.drop(columns=c, inplace=True)
    return out

def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    s = set(df.columns)
    return [c for c in cols if c in s]

def ensure_ticket_id_text(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def _to_text(x):
        if pd.isna(x):
            return ""
        try:
            nx = pd.to_numeric(x, errors="coerce")
            if pd.isna(nx):
                return str(x)
            return str(int(nx))
        except Exception:
            return str(x)
    for c in ("ticket_id", "Номер заявки"):
        if c in out.columns:
            out[c] = out[c].apply(_to_text)
    return out

def add_totals_row_numeric(df: pd.DataFrame, label="ИТОГО") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and is_amount_col(c)]
    if not num_cols:
        return out
    totals = {c: out[c].sum(skipna=True) for c in num_cols}
    label_col = next((c for c in ["plant_name","service_group","warranty_type","ticket_id","Номер заявки"] if c in out.columns), None)
    row = {c: "" for c in out.columns}
    row.update(totals)
    if label_col:
        row[label_col] = label
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)

def format_numbers_ru(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in out.columns:
        if c in EXCLUDE_NUM_FMT:
            continue
        if str(c).lower().endswith("id") or "id_" in str(c).lower():
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(
                lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
                if pd.notnull(x) else ""
            )
    return out

def write_sheets_to_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_money      = wb.add_format({'num_format': MONEY_FORMAT_RU})
        fmt_text       = wb.add_format({'num_format': '@'})
        fmt_bold       = wb.add_format({'bold': True})
        fmt_bold_money = wb.add_format({'bold': True, 'num_format': MONEY_FORMAT_RU})

        for name, df in sheets.items():
            dfw = ensure_ticket_id_text(df)
            dfw.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]

            money_cols_idx = []
            for i, c in enumerate(dfw.columns):
                if is_amount_col(c) and pd.api.types.is_numeric_dtype(dfw[c]):
                    money_cols_idx.append(i)

            # base formats
            for i, c in enumerate(dfw.columns):
                if c in ("ticket_id","Номер заявки"):
                    ws.set_column(i, i, 18, fmt_text)
                elif i in money_cols_idx:
                    ws.set_column(i, i, 16, fmt_money)
                else:
                    ws.set_column(i, i, 12)

            # autofit with format preservation
            for i, c in enumerate(dfw.columns):
                vals = dfw.iloc[:, i].astype(str).head(200).tolist()
                width = min(max(10, max(len(str(c)), *(len(v) for v in vals)) + 2), 42)
                keep = fmt_text if c in ("ticket_id","Номер заявки") else (fmt_money if i in money_cols_idx else None)
                ws.set_column(i, i, width, keep)

            # header freeze + filter
            ws.freeze_panes(1, 0)
            ws.autofilter(f"A1:{_colnum_to_excel(len(dfw.columns)-1)}1")

            # totals row
            nrows = len(dfw)
            if nrows > 0 and money_cols_idx:
                tr = nrows + 1
                label_col = next((j for j in range(len(dfw.columns)) if j not in money_cols_idx), 0)
                ws.write(tr, label_col, "ИТОГО", fmt_bold)
                for j in money_cols_idx:
                    col = _colnum_to_excel(j)
                    ws.write_formula(tr, j, f"=SUM({col}2:{col}{nrows+1})", fmt_bold_money)

    output.seek(0)
    return output.getvalue()

# ---------- Config ----------
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

# ---------- Sidebar ----------
st.sidebar.header("Настройки входных данных")
sheet_name = st.sidebar.text_input("Имя листа в Excel", value="Доставленный")
excluded_types = st.sidebar.text_input("Исключить из затрат завода (через запятую)", value="ВЫЗОВ, ПРОДАЖА")
sp_labels = st.sidebar.text_input("Метки запчастей (через запятую)", value="Запчасть")

st.sidebar.markdown("---")
st.sidebar.subheader("Курсы валют и ML скидок")
rates_file = st.sidebar.file_uploader("Курс UZS→USD (CSV/XLSX)", type=["csv","xlsx"])
ml_training_file = st.sidebar.file_uploader("История скидок с метками (CSV/XLSX)", type=["csv","xlsx"])
ml_threshold = st.sidebar.slider("Порог уверенности ML (скидки):", 0.5, 0.99, 0.85, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Загрузите CRM выгрузку")
uploaded = st.sidebar.file_uploader("Excel из CRM (.xlsx)", type=["xlsx"])

apply_filters_to_export = st.sidebar.checkbox("Применить фильтры к экспортируемому Excel", value=False)

# ---------- Cached I/O ----------
@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes, sheet: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)

@st.cache_data(show_spinner=False)
def read_rates(file) -> pd.DataFrame:
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    cols = {c.strip().lower(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if k in ("date","дата")), None)
    rate_col = next((cols[k] for k in cols if k in ("rate","курс","uzs_usd","uzs_to_usd")), None)
    if not date_col or not rate_col:
        raise ValueError("В курсе валют нужны колонки 'date/Дата' и 'rate/Курс'.")
    df = df.rename(columns={date_col: "date", rate_col: "rate"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["date","rate"]).sort_values("date").drop_duplicates("date", keep="last")
    return df[["date","rate"]]

@st.cache_data(show_spinner=False)
def read_training_discounts(file) -> pd.DataFrame:
    df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)
    low = {c.strip().lower(): c for c in df.columns}
    desc_col = next((low[k] for k in ("description","описание","discount_desc","desc","основание","комментарий") if k in low), None)
    label_col = next((low[k] for k in ("approved_by","утверждено","label","метка","source","responsible") if k in low), None)
    if not desc_col or not label_col:
        raise ValueError("В обучающем датасете нужны колонки 'description/описание' и 'approved_by/утверждено'.")
    df = df.rename(columns={desc_col:"description", label_col:"approved_by"})
    df["description"] = df["description"].astype(str).fillna("")
    df["approved_by"] = df["approved_by"].astype(str).str.strip().str.upper()
    label_map = {"ЗАВОД":"PLANT","PLANT":"PLANT","СЦ":"SC","SERVICE CENTER":"SC","SC":"SC",
                 "ДБЛОК":"SP","СП":"SP","SPARE PARTS":"SP","D-BLOCK":"SP"}
    df["approved_by"] = df["approved_by"].map(lambda x: label_map.get(x, x))
    df = df.loc[df["description"].str.strip().ne("")].drop_duplicates(subset=["description","approved_by"])
    return df[["description","approved_by"]]

@st.cache_resource(show_spinner=False)
def train_discount_classifier(train_df: pd.DataFrame):
    if not SKLEARN_OK:
        raise RuntimeError(f"ML-модуль недоступен: {SKLEARN_ERR}")
    pipe = SkPipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto"))
    ])
    pipe.fit(train_df["description"].values, train_df["approved_by"].values)
    return pipe

# ---------- Domain logic ----------
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
        "Скидки": "Discounts"
    })
    required_cols: Tuple[str, ...] = (
        "ticket_id","warranty_type","service_group","sum_product","sum_service",
        "discount_service","discount_product","plant_name","material_type",
        "service_code","service_unit_price","qty_service","sps_date"
    )

def load_and_normalize(file_bytes: bytes, cfg: PipelineConfig, sheet_name: Optional[str] = None) -> pd.DataFrame:
    df = read_excel_cached(file_bytes, sheet_name or cfg.src_sheet)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k:v for k,v in cfg.rename_map.items() if k in df.columns})
    df = coalesce_duplicate_columns(df)

    for col in cfg.required_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["sum_product","sum_service","discount_service","discount_product","service_unit_price","qty_service"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["qty_service"] = df["qty_service"].fillna(1)
    df[["sum_product","sum_service","discount_service","discount_product","service_unit_price"]] = \
        df[["sum_product","sum_service","discount_service","discount_product","service_unit_price"]].fillna(0)

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
    out = df.copy()
    out["is_excluded_for_manufacturer"] = out["service_group_up"].isin([s.upper() for s in cfg.excluded_for_manufacturer])
    out["is_spare_part"] = out["material_type"].isin(cfg.spare_parts_labels)
    return out

def apply_latest_service_price(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    elig = (df["warranty_type"].isin(["G1","G2"])) & (~df["is_spare_part"]) & df["service_code"].notna()
    df_elig = df.loc[elig].copy().sort_values(["service_code","sps_date"])

    latest_price_map = (
        df_elig.dropna(subset=["service_unit_price","sps_date"])
               .groupby("service_code")["service_unit_price"].last().to_dict()
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
    audit = audit[audit["delta_sum_service"] != 0]
    return df, audit

def apply_usd_conversion(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    working_date = df["sps_date"]
    if working_date.isna().all():
        for f in ["service_date","post_date","sold_at","created_at"]:
            if f in df.columns:
                wd = pd.to_datetime(df[f], errors="coerce")
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
    base_cols = existing_cols(df, ["plant_name","warranty_type","ticket_id","product_name","product_code"])

    if use_usd:
        vals_src = existing_cols(df, ["sum_product_usd","discount_product_usd"])
        pay_lines = df.loc[mask_pay, base_cols + vals_src].copy()
        pay_lines = pay_lines.rename(columns={
            "sum_product_usd":"sum_product",
            "discount_product_usd":"discount_product"
        })
    else:
        vals_src = existing_cols(df, ["sum_product","discount_product"])
        pay_lines = df.loc[mask_pay, base_cols + vals_src].copy()

    sum_cols_ok = existing_cols(pay_lines, ["sum_product","discount_product"])
    pay_summary = (
        pay_lines.groupby(existing_cols(pay_lines, ["plant_name","warranty_type"]), as_index=False)
        .agg(**{
            "Сумма_запчастей": (sum_cols_ok[0], "sum") if "sum_product" in sum_cols_ok else ("ticket_id", "size"),
            "Скидки_по_запчастям_инфо": (sum_cols_ok[1], "sum") if "discount_product" in sum_cols_ok else ("ticket_id", "size"),
        })
        .sort_values("Сумма_запчастей", ascending=False, na_position="last")
    )

    mask_disc = df["is_spare_part"] & (
        (df.get("discount_product", 0) > 0) | (df.get("discount_service", 0) > 0) |
        (df.get("discount_product_usd", 0) > 0) | (df.get("discount_service_usd", 0) > 0)
    )

    base_cols_disc = existing_cols(df, ["plant_name","warranty_type","ticket_id","product_name","product_code"])
    if use_usd:
        vals_disc = existing_cols(df, ["discount_product_usd","discount_service_usd","sum_product_usd","sum_service_usd","sum_total_usd"])
        disc_register = df.loc[mask_disc, base_cols_disc + vals_disc].copy().rename(columns={
            "discount_product_usd":"discount_product",
            "discount_service_usd":"discount_service",
            "sum_product_usd":"sum_product",
            "sum_service_usd":"sum_service",
            "sum_total_usd":"sum_total"
        })
    else:
        vals_disc = existing_cols(df, ["discount_product","discount_service","sum_product","sum_service","sum_total"])
        disc_register = df.loc[mask_disc, base_cols_disc + vals_disc].copy()

    return pay_lines, pay_summary, disc_register

def build_changed_prices_reports(df_before: pd.DataFrame, audit: pd.DataFrame, cfg: PipelineConfig):
    changed_lines = audit.rename(columns={
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

    if audit.empty:
        changed_summary = pd.DataFrame(columns=["Код услуги","Старая_цена","Новая_цена",
                                                "Последняя дата СПС","Строк_затронуто","Уникальных_заявок","Дельта_сумма"])
        return changed_summary, changed_lines

    pairs = (
        audit.groupby("service_code", as_index=False)
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

# ---------- Filters UI ----------
def apply_ui_filters(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        plants = ["(все)"] + sorted(d["plant_name"].dropna().astype(str).unique().tolist())
        sel = st.selectbox("Фильтр: завод", plants)
        if sel != "(все)":
            d = d[d["plant_name"] == sel]
    with c2:
        brands = ["(все)"] + sorted(d.get("brand", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
        sel = st.selectbox("Фильтр: бренд", brands)
        if sel != "(все)" and "brand" in d.columns:
            d = d[d["brand"] == sel]
    with c3:
        w = st.selectbox("Фильтр: гарантия", ["(все)","G1","G2","G3"])
        if w != "(все)":
            d = d[d["warranty_type"] == w]
    with c4:
        min_d, max_d = pd.to_datetime(d["sps_date"]).min(), pd.to_datetime(d["sps_date"]).max()
        if pd.notna(min_d) and pd.notna(max_d):
            start, end = st.date_input("Период по 'дата спс'", value=(min_d.date(), max_d.date()))
            if start and end:
                mask = (pd.to_datetime(d["sps_date"]).dt.date >= start) & (pd.to_datetime(d["sps_date"]).dt.date <= end)
                d = d[mask]
    return d

# ---------- MAIN ----------
cfg = PipelineConfig(
    src_sheet=sheet_name,
    excluded_for_manufacturer=tuple([s.strip().upper() for s in excluded_types.split(",") if s.strip()]),
    spare_parts_labels=tuple([s.strip() for s in sp_labels.split(",") if s.strip()]),
    ml_threshold=ml_threshold
)

if uploaded is None:
    st.info("Загрузите Excel-файл CRM (.xlsx) в сайдбаре, чтобы начать.")
    st.stop()

try:
    # Load & flags
    with st.spinner("Читаем и нормализуем данные..."):
        df = load_and_normalize(uploaded.getvalue(), cfg, sheet_name=sheet_name)
        df = add_line_flags(df, cfg)
        df_before = df.copy()

    # Price adjust
    with st.spinner("Корректируем цены G1/G2 по последним 'дата спс'..."):
        df, audit_price_adj = apply_latest_service_price(df, cfg)
        changed_summary, changed_lines = build_changed_prices_reports(df_before, audit_price_adj, cfg)

    # USD conversion
    usd_enabled = False
    if rates_file is not None:
        with st.spinner("Применяем курсы UZS→USD..."):
            rates = read_rates(rates_file)
            df = apply_usd_conversion(df, rates)
            usd_enabled = True
    else:
        st.info("Курсы валют не загружены — USD-агрегации будут пропущены.")

    # Preview filters
    st.markdown("### Фильтры предпросмотра")
    df_view = apply_ui_filters(df)

    # Aggregations for VIEW
    with st.spinner("Строим своды (UZS) по фильтрам..."):
        agg_id = aggregate_by_ticket(df_view, use_usd=False)
        warranty_totals = warranty_totals_from_id(agg_id)
        ar_id, ar_summary = ar_by_plant(df_view, use_usd=False)
        g3_lines, g3_summary = g3_views(df_view, use_usd=False)
        dblock_pay_lines, dblock_pay_summary, dblock_disc_register = dblock_outputs(df_view, use_usd=False)

    if usd_enabled:
        with st.spinner("Строим своды (USD) по фильтрам..."):
            agg_id_usd = aggregate_by_ticket(df_view, use_usd=True)
            warranty_totals_usd = warranty_totals_from_id(agg_id_usd)
            ar_id_usd, ar_summary_usd = ar_by_plant(df_view, use_usd=True)
            g3_lines_usd, g3_summary_usd = g3_views(df_view, use_usd=True)
            dblock_pay_lines_usd, dblock_pay_summary_usd, dblock_disc_register_usd = dblock_outputs(df_view, use_usd=True)
    else:
        agg_id_usd = warranty_totals_usd = ar_id_usd = ar_summary_usd = None
        g3_lines_usd = g3_summary_usd = dblock_pay_lines_usd = dblock_pay_summary_usd = dblock_disc_register_usd = None

    # ML (on filtered view)
    confident_ml = review_ml = pd.DataFrame()
    if ml_training_file is not None and SKLEARN_OK:
        with st.spinner("Обучаем модель скидок и применяем к текущим данным..."):
            train_df = read_training_discounts(ml_training_file)
            model = train_discount_classifier(train_df)

            def classify_discounts(current_df: pd.DataFrame):
                dfc = current_df.copy()
                text_col = next((c for c in ["discount_description","описание скидки","описание","description","desc","sps1","sps2","service_name"] if c in dfc.columns), None)
                if text_col is None:
                    return pd.DataFrame(), pd.DataFrame()
                has_disc = (dfc.get("discount_product",0) > 0) | (dfc.get("discount_service",0) > 0)
                cand = dfc.loc[has_disc].copy()
                if cand.empty:
                    return pd.DataFrame(), pd.DataFrame()
                texts = cand[text_col].astype(str).fillna("")
                try:
                    proba = model.predict_proba(texts); classes = model.classes_
                    idx = np.argmax(proba, axis=1)
                    cand["ML_Метка"] = classes[idx]
                    cand["ML_Уверенность"] = proba[np.arange(len(cand)), idx]
                except Exception:
                    cand["ML_Метка"] = model.predict(texts)
                    cand["ML_Уверенность"] = np.nan
                return cand.loc[cand["ML_Уверенность"] >= cfg.ml_threshold], cand.loc[(cand["ML_Уверенность"] < cfg.ml_threshold) | (cand["ML_Уверенность"].isna())]

            confident_ml, review_ml = classify_discounts(df_view)
    elif ml_training_file is None:
        st.info("Не загружен обучающий датасет скидок — классификация пропущена.")
    else:
        st.info(f"ML-модуль недоступен: {SKLEARN_ERR}")

    # Dashboard
    st.success("Готово! Предпросмотр ключевых сводок ниже и кнопка скачивания Excel.")
    st.markdown("### 📊 Дашборд — ключевые показатели")
    k1, k2, k3, k4 = st.columns(4)
    ssum = lambda d,c: float(pd.to_numeric(d.get(c, pd.Series(dtype=float)), errors="coerce").sum())
    k1.metric("Итого оборот (UZS)", f"{ssum(df_view,'sum_total'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    k2.metric("Скидки всего (UZS)", f"{ssum(df_view,'discount_total'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    k3.metric("Услуги (UZS)", f"{ssum(df_view,'sum_service'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    k4.metric("Материалы (UZS)", f"{ssum(df_view,'sum_product'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    if "sum_total_usd" in df_view.columns:
        k5,k6,k7,k8 = st.columns(4)
        k5.metric("Итого оборот (USD)", f"{ssum(df_view,'sum_total_usd'):,.2f}")
        k6.metric("Скидки (USD)", f"{ssum(df_view,'discount_total_usd'):,.2f}")
        k7.metric("Услуги (USD)", f"{ssum(df_view,'sum_service_usd'):,.2f}")
        k8.metric("Материалы (USD)", f"{ssum(df_view,'sum_product_usd'):,.2f}")

    st.markdown("### 📈 Графики")
    if ALTAIR_OK:
        c1, c2 = st.columns(2)
        with c1:
            top_plants = (
                df_view[df_view["warranty_type"].isin(["G1","G2"])]
                .groupby("plant_name", as_index=False)["sum_total"].sum()
                .sort_values("sum_total", ascending=False).head(10)
            )
            if not top_plants.empty:
                st.altair_chart(
                    alt.Chart(top_plants).mark_bar().encode(
                        x=alt.X("sum_total:Q", title="Оборот (UZS)"),
                        y=alt.Y("plant_name:N", sort='-x', title="Завод"),
                        tooltip=["plant_name","sum_total"]
                    ).properties(height=320),
                    use_container_width=True
                )
        with c2:
            g3_grp = (
                df_view[df_view["warranty_type"]=="G3"]
                .groupby("service_group", as_index=False)["sum_total"].sum()
                .sort_values("sum_total", ascending=False).head(10)
            )
            if not g3_grp.empty:
                st.altair_chart(
                    alt.Chart(g3_grp).mark_bar().encode(
                        x=alt.X("sum_total:Q", title="Оборот G3 (UZS)"),
                        y=alt.Y("service_group:N", sort='-x', title="Группа услуг"),
                        tooltip=["service_group","sum_total"]
                    ).properties(height=320),
                    use_container_width=True
                )
        if df_view["sps_date"].notna().any():
            ts = (
                df_view.dropna(subset=["sps_date"])
                       .assign(day=lambda d: pd.to_datetime(d["sps_date"]).dt.date)
                       .groupby("day", as_index=False)["sum_total"].sum()
                       .sort_values("day")
            )
            if not ts.empty:
                st.altair_chart(
                    alt.Chart(ts).mark_line(point=True).encode(
                        x=alt.X("day:T", title="Дата СПС"),
                        y=alt.Y("sum_total:Q", title="Оборот (UZS)"),
                        tooltip=["day","sum_total"]
                    ).properties(height=300),
                    use_container_width=True
                )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Итоги по гарантии (UZS)")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(warranty_totals).head(200)))
        st.subheader("AR по заводам — свод (UZS)")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(ar_summary).head(200)))
        if usd_enabled:
            st.subheader("Итоги по гарантии (USD)")
            st.dataframe(format_numbers_ru(add_totals_row_numeric(warranty_totals_usd).head(200)))
            st.subheader("AR по заводам — свод (USD)")
            st.dataframe(format_numbers_ru(add_totals_row_numeric(ar_summary_usd).head(200)))
    with c2:
        st.subheader("G3 — свод (UZS)")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(g3_summary).head(200)))
        if usd_enabled:
            st.subheader("G3 — свод (USD)")
            st.dataframe(format_numbers_ru(add_totals_row_numeric(g3_summary_usd).head(200)))
        st.subheader("Изменённые цены (свод)")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(changed_summary).head(200)))
        if not confident_ml.empty:
            st.subheader("Скидки (ML, уверенные предсказания)")
            st.dataframe(format_numbers_ru(confident_ml.head(200)))
        if not review_ml.empty:
            st.subheader("Скидки (на ручную проверку)")
            st.dataframe(format_numbers_ru(review_ml.head(200)))

    # Export — from filtered (if checkbox) or full dataset
    base = df_view if apply_filters_to_export else df
    agg_id_e = aggregate_by_ticket(base, use_usd=False)
    warranty_totals_e = warranty_totals_from_id(agg_id_e)
    ar_id_e, ar_summary_e = ar_by_plant(base, use_usd=False)
    g3_lines_e, g3_summary_e = g3_views(base, use_usd=False)
    dblock_pay_lines_e, dblock_pay_summary_e, dblock_disc_register_e = dblock_outputs(base, use_usd=False)

    sheets = {
        "Итоги_по_ID": agg_id_e,
        "Итоги_по_Гарантии": warranty_totals_e,
        "АР_по_Заводам_ID": ar_id_e,
        "АР_по_Заводам_Свод": ar_summary_e,
        "Г3_Строки": g3_lines_e,
        "Г3_Свод": g3_summary_e,
        "ДБлок_Кредиторка_Строки": dblock_pay_lines_e,
        "ДБлок_Кредиторка_Свод": dblock_pay_summary_e,
        "ДБлок_Скидки": dblock_disc_register_e,
        "G1G2_Корректировка_Тарифов": audit_price_adj,
        "G1G2_Измененные_Цены_Свод": changed_summary,
        "G1G2_Измененные_Цены_Строки": changed_lines,
    }

    if usd_enabled:
        agg_id_u = aggregate_by_ticket(base, use_usd=True)
        warranty_totals_u = warranty_totals_from_id(agg_id_u)
        ar_id_u, ar_summary_u = ar_by_plant(base, use_usd=True)
        g3_lines_u, g3_summary_u = g3_views(base, use_usd=True)
        dblock_pay_lines_u, dblock_pay_summary_u, dblock_disc_register_u = dblock_outputs(base, use_usd=True)
        sheets.update({
            "Итоги_по_ID_USD": agg_id_u,
            "Итоги_по_Гарантии_USD": warranty_totals_u,
            "АР_по_Заводам_ID_USD": ar_id_u,
            "АР_по_Заводам_Свод_USD": ar_summary_u,
            "Г3_Строки_USD": g3_lines_u,
            "Г3_Свод_USD": g3_summary_u,
            "ДБлок_Кредиторка_Строки_USD": dblock_pay_lines_u,
            "ДБлок_Кредиторка_Свод_USD": dblock_pay_summary_u,
            "ДБлок_Скидки_USD": dblock_disc_register_u,
        })

    xlsx_bytes = write_sheets_to_bytes(sheets)
    st.download_button(
        "💾 Скачать готовый Excel-отчёт",
        data=xlsx_bytes,
        file_name="Финансовые_итоги_месяца.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

except Exception as e:
    st.error(f"Ошибка: {e}")
