# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple
from pathlib import Path

# ==================== CONFIGURATION ====================

@dataclass
class PipelineConfig:
    src_file: Path
    src_sheet: str = "Доставленный"
    output_xlsx: Path = Path("Финансовые_итоги_месяца.xlsx")

    excluded_for_manufacturer: Tuple[str, ...] = ("ВЫЗОВ", "ПРОДАЖА")
    spare_parts_labels: Tuple[str, ...] = ("Запчасть", )
    ml_threshold: float = 0.85
    workshop_bring_in_keywords: Tuple[str, ...] = (
        "принёс", "принес", "сам доставил", "самовывоз", "мастерская", "приём в сц", "прием в сц"
    )

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

# ==================== HELPERS (columns & formatting) ====================

def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Если после rename появились дубли имён:
    - берём первое непустое слева (bfill по оси 1),
    - оставляем один столбец с целевым именем,
    - остальные дубликаты удаляем.
    """
    out = df.copy()
    counts = out.columns.value_counts()
    dup_names = counts[counts > 1].index.tolist()
    for name in dup_names:
        idxs = [i for i, c in enumerate(out.columns) if c == name]
        cols = [out.columns[i] for i in idxs]
        combined = out[cols].bfill(axis=1).iloc[:, 0]
        out[name] = combined
        for c in cols[1:]:
            out.drop(columns=c, inplace=True)
    return out

def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Вернуть только те имена из cols, которые реально присутствуют в df."""
    s = set(df.columns)
    return [c for c in cols if c in s]

# ==================== LOAD & NORMALIZE ====================

def load_and_normalize(cfg: PipelineConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.src_file, sheet_name=cfg.src_sheet)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in cfg.rename_map.items() if k in df.columns})
    df = coalesce_duplicate_columns(df)  # ← Убираем дубли заголовков сразу

    for col in cfg.required_cols:
        if col not in df.columns:
            df[col] = np.nan

    for col in ["sum_product","sum_service","discount_service","discount_product","service_unit_price","qty_service"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["qty_service"] = df["qty_service"].fillna(1)
    df[["sum_product","sum_service","discount_service","discount_product","service_unit_price"]] = \
        df[["sum_product","sum_service","discount_service","discount_product","service_unit_price"]].fillna(0)

    # даты/тексты
    for dcol in ["sps_date","service_date","post_date","sold_at","created_at"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    df["service_group_up"] = df["service_group"].astype(str).str.upper().str.strip()
    df["warranty_type"] = df["warranty_type"].astype(str).str.upper().str.strip()
    df["material_type"] = df["material_type"].astype(str).str.strip()

    df["sum_total"] = df["sum_product"] + df["sum_service"]
    df["discount_total"] = df["discount_service"] + df["discount_product"]
    return df

# ==================== FLAGS ====================

def add_line_flags(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    df["is_excluded_for_manufacturer"] = df["service_group_up"].isin([s.upper() for s in cfg.excluded_for_manufacturer])
    df["is_spare_part"] = df["material_type"].isin(cfg.spare_parts_labels)
    return df

# ==================== PRICE NORMALIZATION (G1/G2 only) ====================

def apply_latest_service_price(df: pd.DataFrame, cfg: PipelineConfig) -> (pd.DataFrame, pd.DataFrame):
    """
    Для G1/G2 и только для строк-услуг (не запчастей):
      - Находим последнюю цену по 'Код услуги' в рамках данных (по 'дата спс')
      - Применяем её к строкам (пересчёт sum_service = qty * новая_цена, обновляем sum_total)
      - Возвращаем df и аудит с изменениями (только строки, где была дельта)
    """
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
    audit = audit[audit["delta_sum_service"] != 0]
    return df, audit

# ==================== AGGREGATIONS ====================

def aggregate_by_ticket(df: pd.DataFrame) -> pd.DataFrame:
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

def ar_by_plant(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    mask = df["warranty_type"].isin(["G1","G2"]) & (~df["is_excluded_for_manufacturer"])
    ar_lines = df.loc[mask, ["plant_name","warranty_type","ticket_id","sum_product","sum_service","sum_total","discount_total"]]
    ar_id = (
        ar_lines.groupby(["plant_name","warranty_type","ticket_id"], as_index=False)
        .agg(sum_product=("sum_product","sum"),
             sum_service=("sum_service","sum"),
             sum_total=("sum_total","sum"),
             discount_total=("discount_total","sum"))
    )
    ar_summary = (
        ar_id.groupby(["plant_name","warranty_type"], as_index=False)
        .agg(sum_product=("sum_product","sum"),
             sum_service=("sum_service","sum"),
             sum_total=("sum_total","sum"),
             discount_total=("discount_total","sum"))
        .sort_values(["sum_total"], ascending=False)
    )
    return ar_id, ar_summary

def g3_views(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    g3_lines = df.loc[df["warranty_type"]=="G3",
        ["service_group","ticket_id","sum_product","sum_service","sum_total","discount_total","is_spare_part","plant_name"]
    ].copy()
    g3_summary = (
        g3_lines.groupby("service_group", as_index=False)
        .agg(sum_product=("sum_product","sum"),
             sum_service=("sum_service","sum"),
             sum_total=("sum_total","sum"),
             discount_total=("discount_total","sum"))
        .sort_values("sum_total", ascending=False)
    )
    return g3_lines, g3_summary

def dblock_outputs(df: pd.DataFrame, use_usd: bool=False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Д-блок:
      - строки по запчастям G2/G3 -> кредиторка;
      - свод по заводам;
      - реестр скидок по запчастям.
    Функция устойчива к отсутствующим колонкам вроде product_name.
    """
    mask_pay = df["is_spare_part"] & df["warranty_type"].isin(["G2","G3"])

    # Базовые колонки
    base_cols = existing_cols(df, ["plant_name","warranty_type","ticket_id","product_name","product_code"])
    if "ticket_id" not in base_cols:
        # чтобы fallback-агрегации могли работать
        base_cols = ["ticket_id"] + base_cols

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

    # Убедимся, что есть ticket_id
    if "ticket_id" not in pay_lines.columns:
        pay_lines["ticket_id"] = ""

    # Свод по Д-блоку
    keys = existing_cols(pay_lines, ["plant_name","warranty_type"])
    # Если ключей нет — агрегируем всё вместе
    if not keys:
        keys = []

    have_sum_product = "sum_product" in pay_lines.columns
    have_disc_product = "discount_product" in pay_lines.columns

    if keys:
        grp = pay_lines.groupby(keys, as_index=False)
        agg_dict = {}
        if have_sum_product:
            agg_dict["Сумма_запчастей"] = ("sum_product","sum")
        if have_disc_product:
            agg_dict["Скидки_по_запчастям_инфо"] = ("discount_product","sum")
        if not agg_dict:
            # если нет суммируемых колонок — посчитаем количество строк
            agg_dict["Сумма_запчастей"] = ("ticket_id","size")
        pay_summary = grp.agg(**agg_dict).sort_values(list(agg_dict.keys())[0], ascending=False)
    else:
        # без ключей — единая строка
        d = {}
        if have_sum_product:
            d["Сумма_запчастей"] = [pay_lines["sum_product"].sum()]
        if have_disc_product:
            d["Скидки_по_запчастям_инфо"] = [pay_lines["discount_product"].sum()]
        if not d:
            d["Сумма_запчастей"] = [len(pay_lines)]
        pay_summary = pd.DataFrame(d)

    # Реестр скидок по запчастям
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

# ==================== EXCEL HELPERS ====================

MONEY_FORMAT_RU = '# ##0,00'   # формат: 1 234 567,89

def _ensure_ticket_id_text(df: pd.DataFrame) -> pd.DataFrame:
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
    return out

def _write_sheet_with_formats(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
    money_cols=None,
    text_id: bool = True,
    autofit: bool = True,
):
    """
    Устойчивый writer:
      - ticket_id как текст
      - money_cols -> RU формат
      - автоподбор ширины без потери форматов
      - корректно работает при дублирующихся заголовках
    """
    wb = writer.book
    df_to_write = _ensure_ticket_id_text(df) if text_id else df.copy()

    # привести деньги к числу (если вдруг есть строки)
    if money_cols:
        for col_name in money_cols:
            if col_name in df_to_write.columns:
                mask = [c == col_name for c in df_to_write.columns]
                if sum(mask) > 1:
                    # несколько одноимённых колонок — обрабатываем по позициям
                    for idx, name in enumerate(df_to_write.columns):
                        if name == col_name:
                            df_to_write.iloc[:, idx] = pd.to_numeric(df_to_write.iloc[:, idx], errors="coerce")
                else:
                    df_to_write[col_name] = pd.to_numeric(df_to_write[col_name], errors="coerce")

    # запись
    df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
    ws = writer.sheets[sheet_name]

    # форматы
    fmt_money = wb.add_format({'num_format': MONEY_FORMAT_RU})
    fmt_text  = wb.add_format({'num_format': '@'})

    # назначим форматы (по позициям — устойчиво к дубликатам)
    col_formats = {}
    for i, name in enumerate(df_to_write.columns):
        if name == "ticket_id":
            ws.set_column(i, i, 16, fmt_text)
            col_formats[i] = fmt_text

    if money_cols:
        money_set = set(money_cols)
        for i, name in enumerate(df_to_write.columns):
            if name in money_set:
                ws.set_column(i, i, 15, fmt_money)
                col_formats[i] = fmt_money

    # автоподбор ширины (сохраняем предыдущий формат)
    if autofit:
        for i, name in enumerate(df_to_write.columns):
            sample_series = df_to_write.iloc[:, i].head(200).astype(str)
            sample_vals = sample_series.tolist()
            max_len = max(len(str(name)), *(len(s) for s in sample_vals)) if len(sample_vals) else len(str(name))
            width = min(max(10, max_len + 2), 40)
            keep_fmt = col_formats.get(i, None)
            ws.set_column(i, i, width, keep_fmt)

# ==================== SAVE OUTPUTS ====================

def save_outputs_ru(
    cfg: PipelineConfig,
    agg_id: pd.DataFrame,
    warranty_totals: pd.DataFrame,
    ar_id: pd.DataFrame,
    ar_summary: pd.DataFrame,
    g3_lines: pd.DataFrame,
    g3_summary: pd.DataFrame,
    dblock_pay_lines: pd.DataFrame,
    dblock_pay_summary: pd.DataFrame,
    dblock_disc_register: pd.DataFrame,
    audit_price_adj: pd.DataFrame,
    changed_summary: pd.DataFrame,
    changed_lines: pd.DataFrame,
) -> Path:
    out = cfg.output_xlsx

    common_money = ["sum_product","sum_service","sum_total","discount_product","discount_service","discount_total"]
    money_ar     = ["sum_product","sum_service","sum_total","discount_total"]
    money_g3     = ["sum_product","sum_service","sum_total","discount_total"]
    money_dblk_s = ["Сумма_запчастей","Скидки_по_запчастям_инфо"]
    money_audit  = ["old_unit_price","new_unit_price","old_sum_service","new_sum_service","delta_sum_service"]

    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        _write_sheet_with_formats(writer, agg_id, "Итоги_по_ID",
                                  money_cols=[c for c in common_money if c in agg_id.columns],
                                  text_id=True)

        _write_sheet_with_formats(writer, warranty_totals, "Итоги_по_Гарантии",
                                  money_cols=[c for c in common_money if c in warranty_totals.columns],
                                  text_id=False)

        _write_sheet_with_formats(writer, ar_id, "АР_по_Заводам_ID",
                                  money_cols=[c for c in money_ar if c in ar_id.columns],
                                  text_id=True)

        _write_sheet_with_formats(writer, ar_summary, "АР_по_Заводам_Свод",
                                  money_cols=[c for c in money_ar if c in ar_summary.columns],
                                  text_id=False)

        _write_sheet_with_formats(writer, g3_lines, "Г3_Строки",
                                  money_cols=[c for c in money_g3 if c in g3_lines.columns],
                                  text_id=True)

        _write_sheet_with_formats(writer, g3_summary, "Г3_Свод",
                                  money_cols=[c for c in money_g3 if c in g3_summary.columns],
                                  text_id=False)

        _write_sheet_with_formats(writer, dblock_pay_lines, "ДБлок_Кредиторка_Строки",
                                  money_cols=[c for c in ["sum_product","discount_product"] if c in dblock_pay_lines.columns],
                                  text_id=True)

        _write_sheet_with_formats(writer, dblock_pay_summary, "ДБлок_Кредиторка_Свод",
                                  money_cols=[c for c in money_dblk_s if c in dblock_pay_summary.columns],
                                  text_id=False)

        _write_sheet_with_formats(writer, dblock_disc_register, "ДБлок_Скидки",
                                  money_cols=[c for c in ["discount_product","discount_service","sum_product","sum_service","sum_total"]
                                              if c in dblock_disc_register.columns],
                                  text_id=True)

        _write_sheet_with_formats(writer, audit_price_adj, "G1G2_Корректировка_Тарифов",
                                  money_cols=[c for c in money_audit if c in audit_price_adj.columns],
                                  text_id=True)

        # Изменённые цены (свод + строки)
        _write_sheet_with_formats(
            writer, changed_summary, "G1G2_Измененные_Цены_Свод",
            money_cols=[c for c in ["Старая_цена","Новая_цена","Дельта_сумма"] if c in changed_summary.columns],
            text_id=False
        )
        _write_sheet_with_formats(
            writer, changed_lines, "G1G2_Измененные_Цены_Строки",
            money_cols=[c for c in [
                "Старая цена услуги","Новая цена услуги",
                "Старая сумма по услуге","Новая сумма по услуге",
                "Дельта (сумма по услуге)"
            ] if c in changed_lines.columns],
            text_id=True
        )

    return out

# ==================== PIPELINE ====================

def run_pipeline(cfg: PipelineConfig) -> Path:
    df = load_and_normalize(cfg)
    df = add_line_flags(df, cfg)

    # snapshot BEFORE нормализации (для последней даты по коду)
    df_before = df.copy()

    # нормализация цен G1/G2
    df, audit_price_adj = apply_latest_service_price(df, cfg)

    # ===== Изменённые цены: строки =====
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

    # ===== Изменённые цены: свод по коду услуги =====
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
        if len(m):
            return m.iloc[0]
        return s.iloc[-1] if edge == "last" else s.iloc[0]

    if audit_price_adj.empty:
        changed_summary = pd.DataFrame(columns=[
            "Код услуги","Старая_цена","Новая_цена","Последняя дата СПС","Строк_затронуто","Уникальных_заявок","Дельта_сумма"
        ])
    else:
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

    # ===== Остальные своды =====
    agg_id = aggregate_by_ticket(df)
    warranty_totals = warranty_totals_from_id(agg_id)
    ar_id, ar_summary = ar_by_plant(df)
    g3_lines, g3_summary = g3_views(df)
    dblock_pay_lines, dblock_pay_summary, dblock_disc_register = dblock_outputs(df)

    # ===== Сохранение =====
    out = save_outputs_ru(
        cfg,
        agg_id,
        warranty_totals,
        ar_id,
        ar_summary,
        g3_lines,
        g3_summary,
        dblock_pay_lines,
        dblock_pay_summary,
        dblock_disc_register,
        audit_price_adj,
        changed_summary,
        changed_lines
    )
    return out

# ==================== MAIN ====================

if __name__ == "__main__":
    cfg = PipelineConfig(
        # ← УКАЖИТЕ ПРАВИЛЬНЫЕ ПУТИ ПОД ВАШ MAC:
        src_file=Path("/Users/3i-a1-2021-177/Desktop/Service/Machine learning/Доставленый_Сентяборь.xlsx"),
        src_sheet="Доставленный",
        output_xlsx=Path("/Users/3i-a1-2021-177/Desktop/Service/Machine learning/Финансовые_итоги_месяца.xlsx"),
    )
    p = run_pipeline(cfg)
    print(p.resolve())
