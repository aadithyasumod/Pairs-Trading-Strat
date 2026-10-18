from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .data import load_price_csv, require_fresh_price_data
from .live_analysis import _analyze_pair, _build_candidates, _download_prices, _filter_symbols, _read_symbols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATH_FIELDS = ("csv", "symbols_file", "out_csv", "out_json", "results_dir")


@dataclass
class MonitorConfig:
    # Data source: use either csv path OR symbols_file
    csv: str | None = None
    symbols_file: str | None = None
    symbol_column: str | None = "Symbol"
    start: str = "2020-01-01"
    end: str | None = None
    chunk_size: int = 100
    sleep_seconds: float = 0.5

    # Analysis settings
    min_history: int = 252
    min_corr: float = 0.65
    top_neighbors: int = 20
    max_candidates: int = 40000
    coint_p_threshold: float = 0.05
    formation_days: int | None = 252  # recent window for the cointegration screen; null = all history
    top_print: int = 15
    max_data_age_days: int | None = 7

    # Alert thresholds
    alert_zscore: float = 2.0
    alert_confidence: float = 1.0
    health_min_for_alert: float = 0.4
    profitable_max_eg_pvalue: float = 0.05
    profitable_min_health: float = 0.5
    profitable_min_confidence: float = 1.2
    profitable_min_abs_zscore: float = 2.0
    capital_per_idea: float = 100_000.0
    deployable_top_n: int = 10

    # Runtime
    loop_minutes: int = 0
    bell_alert: bool = True
    webhook_url: str | None = None

    # Traffic light thresholds for strategy health score
    traffic_green_min: float = 0.65
    traffic_yellow_min: float = 0.45

    # Outputs
    out_csv: str | None = str(PROJECT_ROOT / "live_analysis_latest.csv")
    out_json: str | None = str(PROJECT_ROOT / "live_analysis_summary.json")
    save_excel_per_run: bool = True
    results_dir: str = str(PROJECT_ROOT / "results")



def _load_config(path: str) -> MonitorConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = json.loads(p.read_text())
    known = {f.name for f in fields(MonitorConfig)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {unknown}")

    # Relative paths in the config are relative to the config file, not the shell's cwd.
    for key in PATH_FIELDS:
        value = data.get(key)
        if value and not Path(value).is_absolute():
            data[key] = str((p.parent / value).resolve())
    return MonitorConfig(**data)



def _fetch_prices(cfg: MonitorConfig) -> pd.DataFrame:
    if cfg.csv:
        return load_price_csv(cfg.csv).prices
    if cfg.symbols_file:
        symbols = _read_symbols(cfg.symbols_file, cfg.symbol_column)
        return _download_prices(symbols, cfg.start, cfg.end, cfg.chunk_size, cfg.sleep_seconds)
    raise RuntimeError("Config must set either 'csv' or 'symbols_file'.")



def _strategy_health(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    avg_health = float(df["health"].mean()) if "health" in df else 0.0
    actionable_rate = float((df["status"] == "ACTIONABLE").mean()) if "status" in df else 0.0
    conf = float(np.clip(df["confidence"].mean() / 3.0, 0.0, 1.0)) if "confidence" in df else 0.0
    score = 0.5 * avg_health + 0.3 * actionable_rate + 0.2 * conf
    return float(np.clip(score, 0.0, 1.0))


def _traffic_light(score: float, cfg: MonitorConfig) -> str:
    if score >= cfg.traffic_green_min:
        return "GREEN"
    if score >= cfg.traffic_yellow_min:
        return "YELLOW"
    return "RED"



def _send_webhook(url: str, payload: dict) -> None:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=8).read()



def _run_once(cfg: MonitorConfig) -> tuple[pd.DataFrame, dict]:
    s_cfg = StrategyConfig(
        min_history=cfg.min_history,
        min_correlation=cfg.min_corr,
        coint_p_threshold=cfg.coint_p_threshold,
        formation_days=cfg.formation_days or 0,
    )

    prices = _fetch_prices(cfg)
    freshness = require_fresh_price_data(prices, cfg.max_data_age_days)
    prices = _filter_symbols(prices, cfg.min_history)
    candidates = _build_candidates(prices, cfg.min_corr, cfg.top_neighbors, cfg.max_candidates)

    rows = []
    for a, b in candidates:
        row = _analyze_pair(prices, a, b, s_cfg, require_screen=True)
        if row:
            rows.append(row)

    if not rows:
        raise RuntimeError("No cointegrated pairs found with current settings.")

    df = pd.DataFrame(rows)
    prio = {"ACTIONABLE": 0, "WATCHLIST": 1, "WATCH_REGIME_WEAK": 2, "NEAR_MEAN": 3, "NO_SIGNAL": 4, "EXTREME_RISK": 5}
    df["priority"] = df["status"].map(prio).fillna(99)
    df = df.sort_values(["priority", "confidence"], ascending=[True, False]).drop(columns=["priority"]).reset_index(drop=True)
    df["estimated_pnl_if_1lakh"] = cfg.capital_per_idea * df["expected_reversion_return"].fillna(0.0)
    df["risk_adjusted_estimated_pnl_if_1lakh"] = (
        df["estimated_pnl_if_1lakh"]
        * np.clip(df["health"], 0.0, 1.0)
        * np.clip(df["confidence"] / 2.0, 0.0, 1.0)
    )
    status_boost = df["status"].map(
        {
            "ACTIONABLE": 1.0,
            "WATCHLIST": 0.8,
            "WATCH_REGIME_WEAK": 0.5,
            "NEAR_MEAN": 0.25,
            "NO_SIGNAL": 0.1,
            "EXTREME_RISK": 0.0,
        }
    ).fillna(0.1)
    df["deployable_score"] = (
        0.45 * np.clip(df["confidence"] / 2.0, 0.0, 1.0)
        + 0.30 * np.clip(df["health"], 0.0, 1.0)
        + 0.15 * np.clip(df["zscore"].abs() / 3.0, 0.0, 1.0)
        + 0.10 * status_boost
    )
    df = df.sort_values(["deployable_score", "risk_adjusted_estimated_pnl_if_1lakh"], ascending=[False, False]).reset_index(drop=True)
    df["deployable_rank"] = np.arange(1, len(df) + 1)
    df["is_deployable"] = df["deployable_rank"] <= max(1, int(cfg.deployable_top_n))
    df["quality_tier"] = np.select(
        [df["deployable_score"] >= 0.75, df["deployable_score"] >= 0.55],
        ["A", "B"],
        default="C",
    )

    def profitability_check(row: pd.Series) -> tuple[bool, str]:
        reasons = []
        ok = True
        if str(row["status"]) != "ACTIONABLE":
            ok = False
            reasons.append("status_not_actionable")
        if float(row["eg_pvalue"]) > cfg.profitable_max_eg_pvalue:
            ok = False
            reasons.append("weak_cointegration")
        if float(row["health"]) < cfg.profitable_min_health:
            ok = False
            reasons.append("low_health")
        if float(row["confidence"]) < cfg.profitable_min_confidence:
            ok = False
            reasons.append("low_confidence")
        if abs(float(row["zscore"])) < cfg.profitable_min_abs_zscore:
            ok = False
            reasons.append("zscore_not_extreme")
        if float(row.get("estimated_pnl_if_1lakh", 0.0)) <= 0:
            ok = False
            reasons.append("low_expected_pnl")
        return ok, "pass" if ok else ",".join(reasons)

    checks = df.apply(profitability_check, axis=1, result_type="expand")
    df["is_profitable_candidate"] = checks[0].astype(bool)
    df["candidate_reasons"] = checks[1].astype(str)

    actionable = df[
        (df["status"] == "ACTIONABLE")
        & (df["health"] >= cfg.health_min_for_alert)
        & (df["confidence"] >= cfg.alert_confidence)
        & (df["zscore"].abs() >= cfg.alert_zscore)
    ]

    health_score = _strategy_health(df)
    summary = {
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "price_data_as_of": freshness.last_observation.isoformat(),
        "price_data_age_days": freshness.age_days,
        "symbols_analyzed": int(prices.shape[1]),
        "candidate_pairs": int(len(candidates)),
        "cointegrated_pairs": int(len(df)),
        "actionable_pairs": int((df["status"] == "ACTIONABLE").sum()),
        "profitable_candidates": int(df["is_profitable_candidate"].sum()),
        "deployable_ideas": int(df["is_deployable"].sum()),
        "alert_pairs": int(len(actionable)),
        "strategy_health_score": health_score,
        "strategy_traffic_light": _traffic_light(health_score, cfg),
        "capital_per_idea": float(cfg.capital_per_idea),
        "deployable_top_n": int(cfg.deployable_top_n),
    }

    if len(df) > 0:
        top = df.iloc[0]
        summary["top_pair"] = f"{top['asset_a']}/{top['asset_b']}"
        summary["top_status"] = str(top["status"])
        summary["top_zscore"] = float(top["zscore"])
        summary["top_confidence"] = float(top["confidence"])

    return df, summary



def _alert_if_needed(cfg: MonitorConfig, df: pd.DataFrame, summary: dict) -> None:
    alert_pairs = df[
        (df["status"] == "ACTIONABLE")
        & (df["health"] >= cfg.health_min_for_alert)
        & (df["confidence"] >= cfg.alert_confidence)
        & (df["zscore"].abs() >= cfg.alert_zscore)
    ]

    if alert_pairs.empty:
        return

    top = alert_pairs.iloc[0]
    msg = (
        f"ALERT: {top['asset_a']}/{top['asset_b']} | {top['suggested_action']} | "
        f"z={top['zscore']:.2f}, health={top['health']:.2f}, conf={top['confidence']:.2f}"
    )
    print("\n" + msg)

    if cfg.bell_alert:
        print("\a", end="")

    webhook = cfg.webhook_url or os.getenv("PAIRS_ALERT_WEBHOOK_URL")
    if webhook:
        payload = {
            "text": msg,
            "summary": summary,
        }
        try:
            _send_webhook(webhook, payload)
            print("Webhook alert sent.")
        except Exception as exc:
            print(f"Webhook alert failed: {exc}")



def _print_report(df: pd.DataFrame, summary: dict, top_print: int) -> None:
    print("\nStrategy Summary")
    for k, v in summary.items():
        print(f"{k}: {v}")

    show = [
        "date",
        "deployable_rank",
        "is_deployable",
        "quality_tier",
        "deployable_score",
        "asset_a",
        "asset_b",
        "status",
        "suggested_action",
        "zscore",
        "health",
        "eg_pvalue",
        "confidence",
        "expected_reversion_return",
        "estimated_pnl_if_1lakh",
        "risk_adjusted_estimated_pnl_if_1lakh",
        "is_profitable_candidate",
        "candidate_reasons",
    ]
    print("\nTop opportunities")
    print(df[show].head(top_print).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


def _next_run_number(results_dir: Path) -> int:
    pat = re.compile(r"^run_(\d+)\.xlsx$")
    max_n = 0
    for p in results_dir.glob("run_*.xlsx"):
        m = pat.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _save_run_excel(cfg: MonitorConfig, df: pd.DataFrame, summary: dict) -> Path:
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_number = _next_run_number(results_dir)
    out_path = results_dir / f"run_{run_number:04d}.xlsx"

    summary_df = pd.DataFrame([summary])
    try:
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="opportunities", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
    except ModuleNotFoundError as exc:
        if "openpyxl" not in str(exc):
            raise
        _write_minimal_xlsx(out_path, {"opportunities": df, "summary": summary_df})

    return out_path


def _xlsx_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<c/>"
    if isinstance(value, (bool, np.bool_)):
        return f"<c t=\"b\"><v>{int(value)}</v></c>"
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return f"<c t=\"n\"><v>{value}</v></c>"
    text = escape(str(value))
    return f"<c t=\"inlineStr\"><is><t>{text}</t></is></c>"


def _worksheet_xml(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows_xml = []
    header_cells = "".join(_xlsx_cell(c) for c in cols)
    rows_xml.append(f"<row r=\"1\">{header_cells}</row>")
    for i, row in enumerate(df.itertuples(index=False), start=2):
        cells = "".join(_xlsx_cell(v) for v in row)
        rows_xml.append(f"<row r=\"{i}\">{cells}</row>")
    sheet_data = "".join(rows_xml)
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
        f"<sheetData>{sheet_data}</sheetData>"
        "</worksheet>"
    )


def _write_minimal_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    # Minimal XLSX package with inline strings; works without openpyxl/xlsxwriter.
    sheet_names = list(sheets.keys())
    workbook_sheets = "".join(
        f"<sheet name=\"{escape(name[:31])}\" sheetId=\"{i+1}\" r:id=\"rId{i+1}\"/>"
        for i, name in enumerate(sheet_names)
    )
    workbook_rels = "".join(
        f"<Relationship Id=\"rId{i+1}\" "
        "Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" "
        f"Target=\"worksheets/sheet{i+1}.xml\"/>"
        for i in range(len(sheet_names))
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            "<Override PartName=\"/xl/workbook.xml\" "
            "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
            + "".join(
                f"<Override PartName=\"/xl/worksheets/sheet{i+1}.xml\" "
                "ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
                for i in range(len(sheet_names))
            )
            + "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" "
            "Target=\"xl/workbook.xml\"/>"
            "</Relationships>",
        )
        zf.writestr(
            "xl/workbook.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            f"<sheets>{workbook_sheets}</sheets>"
            "</workbook>",
        )
        zf.writestr(
            "xl/_rels/workbook.xml.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            f"{workbook_rels}</Relationships>",
        )
        for i, name in enumerate(sheet_names, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", _worksheet_xml(sheets[name]))



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple live pairs monitor (analysis only)")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "monitor_config.json"),
        help="Path to monitor config JSON",
    )
    return p.parse_args()



def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    while True:
        df, summary = _run_once(cfg)
        _print_report(df, summary, cfg.top_print)

        if cfg.out_csv:
            Path(cfg.out_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cfg.out_csv, index=False)

        if cfg.out_json:
            Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.out_json).write_text(json.dumps(summary, indent=2))

        if cfg.save_excel_per_run:
            saved = _save_run_excel(cfg, df, summary)
            print(f"\nSaved run workbook: {saved}")

        _alert_if_needed(cfg, df, summary)

        if cfg.loop_minutes <= 0:
            break

        print(f"\nSleeping for {cfg.loop_minutes} minutes...")
        time.sleep(cfg.loop_minutes * 60)


if __name__ == "__main__":
    main()
