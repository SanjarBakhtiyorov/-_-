# -*- coding: utf-8 -*-
"""
Monthly_pipeline_3.py
Полнофункциональные фильтры + дашборд + выгрузка в Excel.
"""

import io
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List

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
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as e:
    SKLEARN_OK, SKLEARN_ERR = False, str(e)

# ---------- Page ----------
st.set_page_config(page_title="Monthly Pipeline 3 — Artel", layout="wide")
st.title("Финансовый пайплайн (Monthly pipeline 3)")
st.caption("Фильтры → своды → дашборд → Excel")

MONEY_FORMAT_RU = '# ##0,00'
EXCLUDE_NUM_FMT = {"ticket_id","Номер заявки","service_code","Код услуги","product_code","Код продукта"}

# ---------- Helpers ----------
def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    counts = out.columns.value_counts()
    for name in counts[counts > 1].index:
        cols = [c for c in out.columns if c == name]
        out[name] = out[cols].bfill(axis=1).iloc[:, 0]
        for c in cols[1:]:
            out.drop(columns=c, inplace=True)
    return out

def ensure_ticket_id_text_inplace(df: pd.DataFrame) -> None:
    if "ticket_id" in df.columns:
        df["ticket_id"] = pd.to_numeric(df["ticket_id"], errors="coerce") \
            .apply(lambda v: "" if pd.isna(v) else str(int(v)))

def existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    s = set(df.columns); return [c for c in cols if c in s]

def is_amount_col(colname: str) -> bool:
    lc = str(colname).lower()
    return any(k in lc for k in ["sum","total","скидк","сумм","цена","дельта","usd"])

def add_totals_row_numeric(df: pd.DataFrame, label="ИТОГО") -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and is_amount_col(c)]
    if not num_cols: return out
    totals = {c: out[c].sum(skipna=True) for c in num_cols}
    label_col = next((c for c in ["plant_name","service_group","warranty_type","ticket_id","Номер заявки"] if c in out.columns), None)
    row = {c: "" for c in out.columns}; row.update(totals)
    if label_col: row[label_col] = label
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)

def format_numbers_ru(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    for c in out.columns:
        if c in EXCLUDE_NUM_FMT: continue
        if str(c).lower().endswith("id") or "id_" in str(c).lower(): continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].apply(
                lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
                if pd.notnull(x) else ""
            )
    return out

def _col_to_excel(n: int) -> str:
    s=""; n+=1
    while n: n,r=divmod(n-1,26); s=chr(65+r)+s
    return s

def write_sheets_to_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        wb = writer.book
        fmt_money = wb.add_format({'num_format': MONEY_FORMAT_RU})
        fmt_text  = wb.add_format({'num_format': '@'})
        fmt_bold  = wb.add_format({'bold': True})
        fmt_bold_money = wb.add_format({'bold': True, 'num_format': MONEY_FORMAT_RU})

        for name, dfw in sheets.items():
            dfw = dfw.copy()
            ensure_ticket_id_text_inplace(dfw)
            dfw.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]

            money_idx = []
            for i,c in enumerate(dfw.columns):
                if is_amount_col(c) and pd.api.types.is_numeric_dtype(dfw[c]):
                    money_idx.append(i)

            # base widths + formats
            for i,c in enumerate(dfw.columns):
                if c in ("ticket_id","Номер заявки"):
                    ws.set_column(i, i, 18, fmt_text)
                elif i in money_idx:
                    ws.set_column(i, i, 16, fmt_money)
                else:
                    ws.set_column(i, i, 12)

            # simple autofit
            for i,c in enumerate(dfw.columns):
                vals = dfw.iloc[:, i].astype(str).head(200).tolist()
                width = min(max(10, max(len(str(c)), *(len(v) for v in vals)) + 2), 42)
                keep = fmt_text if c in ("ticket_id","Номер заявки") else (fmt_money if i in money_idx else None)
                ws.set_column(i, i, width, keep)

            # totals row
            nrows = len(dfw)
            if nrows > 0 and money_idx:
                tr = nrows + 1
                label_col = next((j for j in range(len(dfw.columns)) if j not in money_idx), 0)
                ws.write(tr, label_col, "ИТОГО", fmt_bold)
                for j in money_idx:
                    col = _col_to_excel(j)
                    ws.write_formula(tr, j, f"=SUM({col}2:{col}{nrows+1})", fmt_bold_money)

    output.seek(0)
    return output.getvalue()

# ---------- Data config ----------
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

# ---------- Sidebar (inputs) ----------
st.sidebar.header("Загрузите данные")
uploaded = st.sidebar.file_uploader("Excel из CRM (.xlsx)", type=["xlsx"])
sheet_name = st.sidebar.text_input("Имя листа", value="Доставленный")

st.sidebar.markdown("---")
st.sidebar.header("Курсы валют (опционально)")
rates_file = st.sidebar.file_uploader("Курс UZS→USD (CSV/XLSX)", type=["csv","xlsx"])

st.sidebar.markdown("---")
apply_filters_to_export = st.sidebar.checkbox("Применить фильтры к Excel-выгрузке", value=True)

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
        raise ValueError("Нужны колонки 'date/Дата' и 'rate/Курс'.")
    df = df.rename(columns={date_col:"date", rate_col:"rate"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    return df.dropna(subset=["date","rate"]).sort_values("date").drop_duplicates("date", keep="last")[["date","rate"]]

# ---------- Pipeline functions ----------
def load_and_normalize(file_bytes: bytes, cfg: PipelineConfig, sheet: Optional[str]) -> pd.DataFrame:
    df = read_excel_cached(file_bytes, sheet or cfg.src_sheet)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k:v for k,v in cfg.rename_map.items() if k in df.columns})
    df = coalesce_duplicate_columns(df)

    for col in cfg.required_cols:
        if col not in df.columns: df[col] = np.nan

    num_cols = ["sum_product","sum_service","discount_service","discount_product","service_unit_price","qty_service"]
    for c in num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["qty_service"] = df["qty_service"].fillna(1)
    df[num_cols[:-1]] = df[num_cols[:-1]].fillna(0)  # все кроме qty

    # dates
    for d in ["sps_date","service_date","post_date","sold_at","created_at"]:
        if d in df.columns: df[d] = pd.to_datetime(df[d], errors="coerce")

    df["service_group_up"] = df["service_group"].astype(str).str.upper().str.strip()
    df["warranty_type"] = df["warranty_type"].astype(str).str.upper().str.strip()
    df["material_type"] = df["material_type"].astype(str).str.strip()

    df["sum_total"] = df["sum_product"] + df["sum_service"]
    df["discount_total"] = df["discount_service"] + df["discount_product"]

    ensure_ticket_id_text_inplace(df)
    return df

def apply_usd_conversion(df: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wd = out["sps_date"]
    if wd.isna().all():
        for f in ["service_date","post_date","sold_at","created_at"]:
            if f in out.columns:
                tmp = pd.to_datetime(out[f], errors="coerce")
                if tmp.notna().any():
                    wd = tmp; break
    out["_rate_date"] = pd.to_datetime(wd, errors="coerce")
    tmp = out[["_rate_date"]].copy()
    tmp["__row__"] = np.arange(len(tmp))
    tmp = tmp.sort_values("_rate_date")
    r = rates.sort_values("date")
    merged = pd.merge_asof(tmp, r, left_on="_rate_date", right_on="date", direction="backward").sort_values("__row__")
    out["usd_rate"] = merged["rate"].values
    for c in ["sum_product","sum_service","sum_total","discount_product","discount_service","discount_total"]:
        if c in out.columns:
            out[c+"_usd"] = np.where(out["usd_rate"] > 0, out[c] / out["usd_rate"], np.nan)
    return out

# ---------- FILTERS (full) ----------
def build_filters(df: pd.DataFrame):
    """Render filters and return filtered DataFrame."""
    d = df.copy()

    st.subheader("Фильтры")

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        plants = ["(все)"] + sorted(d.get("plant_name", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
        plant_sel = st.selectbox("Завод", plants, index=0)
        if plant_sel != "(все)":
            d = d[d["plant_name"] == plant_sel]

    with r1c2:
        brands = ["(все)"] + sorted(d.get("brand", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
        brand_sel = st.selectbox("Бренд", brands, index=0)
        if brand_sel != "(все)" and "brand" in d.columns:
            d = d[d["brand"] == brand_sel]

    with r1c3:
        warranties = ["(все)", "G1", "G2", "G3"]
        w_sel = st.selectbox("Гарантия", warranties, index=0)
        if w_sel != "(все)":
            d = d[d["warranty_type"] == w_sel]

    with r1c4:
        mind, maxd = pd.to_datetime(d["sps_date"]).min(), pd.to_datetime(d["sps_date"]).max()
        if pd.notna(mind) and pd.notna(maxd):
            start, end = st.date_input("Период по 'дата спс'", value=(mind.date(), maxd.date()))
            if start and end:
                mask = (pd.to_datetime(d["sps_date"]).dt.date >= start) & (pd.to_datetime(d["sps_date"]).dt.date <= end)
                d = d[mask]

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        all_groups = sorted(d.get("service_group", pd.Series([], dtype=str)).dropna().astype(str).unique().tolist())
        sel_groups = st.multiselect("Группы услуг", all_groups, default=all_groups[:10] if len(all_groups)>10 else all_groups)
        if sel_groups:
            d = d[d["service_group"].astype(str).isin(sel_groups)]

    with r2c2:
        part_mode = st.select_slider("Запчасти", options=["Все", "Только запчасти", "Исключить запчасти"], value="Все")
        if "is_spare_part" in d.columns:
            if part_mode == "Только запчасти":
                d = d[d["is_spare_part"] == True]
            elif part_mode == "Исключить запчасти":
                d = d[d["is_spare_part"] == False]

    with r2c3:
        min_sum, max_sum = float(d["sum_total"].min() or 0), float(d["sum_total"].max() or 0)
        min_v, max_v = st.slider("Оборот (UZS), диапазон", min_value=0.0, max_value=max(1.0, max_sum), value=(0.0, max_sum))
        d = d[(d["sum_total"] >= min_v) & (d["sum_total"] <= max_v)]

    with r2c4:
        min_disc, max_disc = float(d["discount_total"].min() or 0), float(d["discount_total"].max() or 0)
        min_dv, max_dv = st.slider("Скидка (UZS), диапазон", min_value=min(0.0, min_disc), max_value=max(1.0, max_disc), value=(min(0.0, min_disc), max_disc))
        d = d[(d["discount_total"] >= min_dv) & (d["discount_total"] <= max_dv)]

    r3c1, r3c2 = st.columns([2,2])
    with r3c1:
        search_text = st.text_input("Поиск по тексту (код/услуга/продукт/комментарий)", value="")
        if search_text.strip():
            patt = search_text.strip().lower()
            search_cols = existing_cols(d, ["service_name","service_group","product_name","product_code","service_code","ticket_id","sps1","sps2"])
            if search_cols:
                mask = False
                for c in search_cols:
                    mask = mask | d[c].astype(str).str.lower().str.contains(patt, na=False)
                d = d[mask]

    with r3c2:
        only_g12_for_ar = st.checkbox("Исключить из затрат завода (ВЫЗОВ/ПРОДАЖА)", value=False)
        if only_g12_for_ar and "service_group_up" in d.columns:
            d = d[~d["service_group_up"].isin(["ВЫЗОВ","ПРОДАЖА"])]

    return d

# ---------- Aggregations ----------
def aggregate_by_ticket(df: pd.DataFrame, use_usd=False) -> pd.DataFrame:
    if use_usd:
        return (df.groupby("ticket_id").agg(
            sum_product=("sum_product_usd","sum"),
            sum_service=("sum_service_usd","sum"),
            sum_total=("sum_total_usd","sum"),
            discount_product=("discount_product_usd","sum"),
            discount_service=("discount_service_usd","sum"),
            discount_total=("discount_total_usd","sum"),
            warranty_type=("warranty_type","first"),
            plant_name=("plant_name","first")
        ).reset_index())
    return (df.groupby("ticket_id").agg(
        sum_product=("sum_product","sum"),
        sum_service=("sum_service","sum"),
        sum_total=("sum_total","sum"),
        discount_product=("discount_product","sum"),
        discount_service=("discount_service","sum"),
        discount_total=("discount_total","sum"),
        warranty_type=("warranty_type","first"),
        plant_name=("plant_name","first")
    ).reset_index())

def warranty_totals_from_id(agg_id: pd.DataFrame) -> pd.DataFrame:
    return (agg_id.groupby("warranty_type")[["sum_product","sum_service","sum_total","discount_total"]].sum().reset_index())

def ar_by_plant(df: pd.DataFrame, use_usd=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["warranty_type"].isin(["G1","G2"]) & (~df.get("is_excluded_for_manufacturer", False).fillna(False))
    cols = ["plant_name","warranty_type","ticket_id"]
    vals = ["sum_product_usd","sum_service_usd","sum_total_usd","discount_total_usd"] if use_usd else \
           ["sum_product","sum_service","sum_total","discount_total"]
    ar_lines = df.loc[mask, cols + vals]
    ar_id = (ar_lines.groupby(["plant_name","warranty_type","ticket_id"], as_index=False)
             .agg(**{vals[0]:(vals[0],"sum"), vals[1]:(vals[1],"sum"), vals[2]:(vals[2],"sum"), vals[3]:(vals[3],"sum")}))
    ar_summary = (ar_id.groupby(["plant_name","warranty_type"], as_index=False)
                  .agg(**{vals[0]:(vals[0],"sum"), vals[1]:(vals[1],"sum"), vals[2]:(vals[2],"sum"), vals[3]:(vals[3],"sum")})
                  .sort_values(vals[2], ascending=False))
    return ar_id, ar_summary

def g3_views(df: pd.DataFrame, use_usd=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = ["service_group","ticket_id","is_spare_part","plant_name"]
    vals = ["sum_product_usd","sum_service_usd","sum_total_usd","discount_total_usd"] if use_usd else \
           ["sum_product","sum_service","sum_total","discount_total"]
    g3_lines = df.loc[df["warranty_type"]=="G3", base+vals].copy()
    g3_summary = (g3_lines.groupby("service_group", as_index=False)
                  .agg(**{vals[0]:(vals[0],"sum"), vals[1]:(vals[1],"sum"), vals[2]:(vals[2],"sum"), vals[3]:(vals[3],"sum")})
                  .sort_values(vals[2], ascending=False))
    return g3_lines, g3_summary

def dblock_outputs(df: pd.DataFrame, use_usd=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask_pay = df["is_spare_part"] & df["warranty_type"].isin(["G2","G3"])
    base = existing_cols(df, ["plant_name","warranty_type","ticket_id","product_name","product_code"])
    if use_usd:
        vals = existing_cols(df, ["sum_product_usd","discount_product_usd"])
        pay_lines = df.loc[mask_pay, base+vals].copy().rename(columns={
            "sum_product_usd":"sum_product","discount_product_usd":"discount_product"
        })
    else:
        vals = existing_cols(df, ["sum_product","discount_product"])
        pay_lines = df.loc[mask_pay, base+vals].copy()
    pay_summary = (pay_lines.groupby(existing_cols(pay_lines, ["plant_name","warranty_type"]), as_index=False)
                   .agg(Сумма_запчастей=("sum_product","sum"),
                        Скидки_по_запчастям_инфо=("discount_product","sum"))
                   .sort_values("Сумма_запчастей", ascending=False, na_position="last"))
    mask_disc = df["is_spare_part"] & (
        (df.get("discount_product",0)>0)|(df.get("discount_service",0)>0)|
        (df.get("discount_product_usd",0)>0)|(df.get("discount_service_usd",0)>0)
    )
    base_d = existing_cols(df, ["plant_name","warranty_type","ticket_id","product_name","product_code"])
    if use_usd:
        vals_d = existing_cols(df, ["discount_product_usd","discount_service_usd","sum_product_usd","sum_service_usd","sum_total_usd"])
        disc = df.loc[mask_disc, base_d+vals_d].copy().rename(columns={
            "discount_product_usd":"discount_product","discount_service_usd":"discount_service",
            "sum_product_usd":"sum_product","sum_service_usd":"sum_service","sum_total_usd":"sum_total"
        })
    else:
        vals_d = existing_cols(df, ["discount_product","discount_service","sum_product","sum_service","sum_total"])
        disc = df.loc[mask_disc, base_d+vals_d].copy()
    return pay_lines, pay_summary, disc

# ---------- MAIN ----------
cfg = PipelineConfig(src_sheet=sheet_name)

if uploaded is None:
    st.info("Загрузите Excel-файл CRM (.xlsx) в сайдбаре.")
    st.stop()

try:
    with st.spinner("Чтение и нормализация..."):
        df = load_and_normalize(uploaded.getvalue(), cfg, sheet_name)

    # опционально — конвертация в USD
    usd_enabled = False
    if rates_file is not None:
        with st.spinner("Применяем курс UZS→USD..."):
            rates = read_rates(rates_file)
            df = apply_usd_conversion(df, rates)
            usd_enabled = True

    # --------- ФИЛЬТРЫ ---------
    df_view = build_filters(df)

    # --------- KPI ---------
    st.success("Готово! Ниже дашборд, своды и выгрузка.")
    st.markdown("### 📊 Ключевые показатели (по отфильтрованным данным)")
    k1,k2,k3,k4 = st.columns(4)
    ssum = lambda d,c: float(pd.to_numeric(d.get(c, pd.Series(dtype=float)), errors="coerce").sum())
    k1.metric("Итого оборот (UZS)", f"{ssum(df_view,'sum_total'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    k2.metric("Скидки (UZS)", f"{ssum(df_view,'discount_total'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    k3.metric("Услуги (UZS)", f"{ssum(df_view,'sum_service'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    k4.metric("Материалы (UZS)", f"{ssum(df_view,'sum_product'):,.2f}".replace(",", "X").replace(".", ",").replace("X", " "))
    if usd_enabled and "sum_total_usd" in df_view.columns:
        k5,k6,k7,k8 = st.columns(4)
        k5.metric("Оборот (USD)", f"{ssum(df_view,'sum_total_usd'):,.2f}")
        k6.metric("Скидки (USD)", f"{ssum(df_view,'discount_total_usd'):,.2f}")
        k7.metric("Услуги (USD)", f"{ssum(df_view,'sum_service_usd'):,.2f}")
        k8.metric("Материалы (USD)", f"{ssum(df_view,'sum_product_usd'):,.2f}")

    # --------- Графики ---------
    st.markdown("### 📈 Графики")
    if ALTAIR_OK:
        cc1, cc2 = st.columns(2)
        with cc1:
            top_plants = (df_view.groupby("plant_name", as_index=False)["sum_total"].sum()
                          .sort_values("sum_total", ascending=False).head(10))
            if not top_plants.empty:
                st.altair_chart(
                    alt.Chart(top_plants).mark_bar().encode(
                        x=alt.X("sum_total:Q", title="Оборот (UZS)"),
                        y=alt.Y("plant_name:N", sort='-x', title="Завод"),
                        tooltip=["plant_name","sum_total"]
                    ).properties(height=320),
                    use_container_width=True
                )
        with cc2:
            g3_grp = (df_view[df_view["warranty_type"]=="G3"]
                      .groupby("service_group",as_index=False)["sum_total"].sum()
                      .sort_values("sum_total", ascending=False).head(10))
            if not g3_grp.empty:
                st.altair_chart(
                    alt.Chart(g3_grp).mark_bar().encode(
                        x=alt.X("sum_total:Q", title="G3 оборот (UZS)"),
                        y=alt.Y("service_group:N", sort='-x', title="Группа услуг"),
                        tooltip=["service_group","sum_total"]
                    ).properties(height=320),
                    use_container_width=True
                )
    else:
        st.info("Altair недоступен, графики отключены.")

    # --------- Своды/таблицы ---------
    with st.spinner("Строим своды..."):
        agg_id = aggregate_by_ticket(df_view, use_usd=False)
        warranty_totals = warranty_totals_from_id(agg_id)
        ar_id, ar_summary = ar_by_plant(df_view, use_usd=False)
        g3_lines, g3_summary = g3_views(df_view, use_usd=False)
        dblock_pay_lines, dblock_pay_summary, dblock_disc_register = dblock_outputs(df_view, use_usd=False)

        if usd_enabled:
            agg_id_usd = aggregate_by_ticket(df_view, use_usd=True)
            warranty_totals_usd = warranty_totals_from_id(agg_id_usd)
            ar_id_usd, ar_summary_usd = ar_by_plant(df_view, use_usd=True)
            g3_lines_usd, g3_summary_usd = g3_views(df_view, use_usd=True)
            dblock_pay_lines_usd, dblock_pay_summary_usd, dblock_disc_register_usd = dblock_outputs(df_view, use_usd=True)
        else:
            warranty_totals_usd = ar_summary_usd = g3_summary_usd = None

    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Итоги по гарантии (UZS)")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(warranty_totals).head(200)))
        st.subheader("AR по заводам — свод (UZS)")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(ar_summary).head(200)))
        if usd_enabled and warranty_totals_usd is not None:
            st.subheader("Итоги по гарантии (USD)")
            st.dataframe(format_numbers_ru(add_totals_row_numeric(warranty_totals_usd).head(200)))
            st.subheader("AR по заводам — свод (USD)")
            st.dataframe(format_numbers_ru(add_totals_row_numeric(ar_summary_usd).head(200)))
    with c2:
        st.subheader("G3 — свод (UZS)")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(g3_summary).head(200)))
        if usd_enabled and g3_summary_usd is not None:
            st.subheader("G3 — свод (USD)")
            st.dataframe(format_numbers_ru(add_totals_row_numeric(g3_summary_usd).head(200)))
        st.subheader("Д-Блок — свод")
        st.dataframe(format_numbers_ru(add_totals_row_numeric(dblock_pay_summary).head(200)))

    # --------- Export ---------
    st.markdown("---")
    st.subheader("Экспорт в Excel")
    base = df_view if apply_filters_to_export else df
    # пересчёт сводов для экспорта (из выбранной базы)
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
    }
    if usd_enabled and "sum_total_usd" in base.columns:
        agg_id_ue = aggregate_by_ticket(base, use_usd=True)
        warranty_totals_ue = warranty_totals_from_id(agg_id_ue)
        ar_id_ue, ar_summary_ue = ar_by_plant(base, use_usd=True)
        g3_lines_ue, g3_summary_ue = g3_views(base, use_usd=True)
        dblock_pay_lines_ue, dblock_pay_summary_ue, dblock_disc_register_ue = dblock_outputs(base, use_usd=True)
        sheets.update({
            "Итоги_по_ID_USD": agg_id_ue,
            "Итоги_по_Гарантии_USD": warranty_totals_ue,
            "АР_по_Заводам_ID_USD": ar_id_ue,
            "АР_по_Заводам_Свод_USD": ar_summary_ue,
            "Г3_Строки_USD": g3_lines_ue,
            "Г3_Свод_USD": g3_summary_ue,
            "ДБлок_Кредиторка_Строки_USD": dblock_pay_lines_ue,
            "ДБлок_Кредиторка_Свод_USD": dblock_pay_summary_ue,
            "ДБлок_Скидки_USD": dblock_disc_register_ue,
        })

    xlsx = write_sheets_to_bytes(sheets)
    st.download_button(
        label="💾 Скачать Excel",
        data=xlsx,
        file_name="Финансовые_итоги_месяца.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

except Exception as e:
    st.error(f"Ошибка: {e}")
