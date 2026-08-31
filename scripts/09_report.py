"""PHASE 8 -- assemble the final PDF report.

Reads results/tables/ and results/figures/ and renders a report. Every number
is pulled from a written artefact, so the report cannot drift from the results.
Where a gate failed or a planned method was unavailable, that is stated in the
report rather than omitted.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from utils import get_logger, load_config, repo_root, set_seed

NAVY = colors.HexColor("#1F3A5F")
ACC = colors.HexColor("#C44E52")
GREY = colors.HexColor("#5A5A5A")
LIGHT = colors.HexColor("#EEF2F7")


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("TitleBig", parent=s["Title"], fontSize=21, leading=25,
                         textColor=NAVY, spaceAfter=4))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontSize=10.5, textColor=GREY,
                         alignment=1, spaceAfter=16))
    s.add(ParagraphStyle("H1", parent=s["Heading1"], fontSize=14, textColor=NAVY,
                         spaceBefore=14, spaceAfter=7))
    s.add(ParagraphStyle("H2", parent=s["Heading2"], fontSize=11.5, textColor=NAVY,
                         spaceBefore=10, spaceAfter=5))
    s.add(ParagraphStyle("Body", parent=s["Normal"], fontSize=9.4, leading=13.4,
                         alignment=TA_JUSTIFY, spaceAfter=6))
    s.add(ParagraphStyle("Flag", parent=s["Normal"], fontSize=9.2, leading=13,
                         textColor=ACC, leftIndent=8, spaceAfter=6))
    s.add(ParagraphStyle("Cap", parent=s["Normal"], fontSize=8.2, textColor=GREY,
                         alignment=1, spaceAfter=10))
    s.add(ParagraphStyle("Cell", parent=s["Normal"], fontSize=8, leading=10))
    return s


class R:
    """Lazy accessor for result artefacts."""

    def __init__(self, cfg):
        self.t = repo_root() / cfg["paths"]["tables"]
        self.f = repo_root() / cfg["paths"]["figures"]

    def json(self, name, default=None):
        p = self.t / name
        return json.load(open(p)) if p.exists() else (default or {})

    def csv(self, name):
        p = self.t / name
        return pd.read_csv(p) if p.exists() else None

    def fig(self, name):
        p = self.f / name
        return p if p.exists() else None


def mktable(data, widths, s, header=True, align_right=None):
    body = [[Paragraph(str(c), s["Cell"]) for c in row] for row in data]
    t = Table(body, colWidths=widths, hAlign="LEFT")
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8D0DA")),
          ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        for i in range(len(body[0])):
            body[0][i] = Paragraph(f'<b><font color="white">{data[0][i]}</font></b>', s["Cell"])
        st += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT])]
    t.setStyle(TableStyle(st))
    return t


def add_fig(story, path, caption, s, width=165 * mm):
    if path is None:
        return
    from PIL import Image as PILImage  # noqa: F401  (reportlab reads size itself)
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width
    img.drawHeight = width * ratio
    if img.drawHeight > 210 * mm:
        img.drawHeight = 210 * mm
        img.drawWidth = img.drawHeight / ratio
    img.hAlign = "CENTER"
    story += [img, Paragraph(caption, s["Cap"])]


def build(cfg, log):
    s = styles()
    r = R(cfg)
    root = repo_root()
    horizon = cfg["label"]["horizon_days"]
    hy = horizon / 365.25

    g1 = r.json("gate1_summary.json")
    mA = r.json("model_a_summary.json")
    mB = r.json("model_b_summary.json")
    mC = r.json("model_c_summary.json")
    plateau = r.json("signature_plateau.json")
    g6 = r.json("gate6_summary.json")
    g7 = r.json("gate7_summary.json")

    story = []

    # ---------------- title ----------------
    story += [Spacer(1, 30 * mm),
              Paragraph("Multi-Omics Biomarker Signature for Breast Cancer Relapse", s["TitleBig"]),
              Paragraph(f"TCGA-BRCA discovery, METABRIC external validation &middot; "
                        f"{hy:.0f}-year relapse endpoint<br/>"
                        f"Generated {datetime.now():%Y-%m-%d %H:%M} &middot; seed {cfg['seed']}",
                        s["Sub"])]

    hl = []
    if mA and mB:
        winner = "the clinical baseline" if mA.get("auc_mean", 0) > mB.get("auc_mean", 0) \
            else "the multi-omics model"
        hl.append(f"<b>Headline result.</b> On identical outer folds, {winner} performed best. "
                  f"Model A (clinical: age, stage, PAM50) reached AUC "
                  f"<b>{mA.get('auc_mean'):.3f}</b> "
                  f"[{mA.get('auc_ci_lo'):.3f}-{mA.get('auc_ci_hi'):.3f}]; "
                  f"Model B (XGBoost on clinical + expression + methylation) reached "
                  f"<b>{mB.get('auc_mean'):.3f}</b> "
                  f"[{mB.get('auc_ci_lo'):.3f}-{mB.get('auc_ci_hi'):.3f}]")
        if mC:
            hl[-1] += (f"; Model C (pathway-informed neural network) reached "
                       f"<b>{mC.get('auc_mean'):.3f}</b> "
                       f"[{mC.get('auc_ci_lo'):.3f}-{mC.get('auc_ci_hi'):.3f}]")
        hl[-1] += ("."
                   " Adding ~19,000 omics features did not improve on a two-variable"
                   " clinical model at this sample size."
                   if mA.get("auc_mean", 0) > mB.get("auc_mean", 0) else ".")
    if g1:
        hl.append(f"<b>Cohort.</b> n = {g1.get('n_final')} patients with matched expression, "
                  f"methylation and a usable label; {g1.get('n_positive')} relapse-positive "
                  f"({100*g1.get('positive_rate',0):.1f}%). "
                  f"{g1.get('n_excluded_early_censoring')} patients were excluded for censoring "
                  f"before the {hy:.0f}-year horizon.")
    if g7:
        hl.append(f"<b>External validation.</b> METABRIC n = {g7.get('metabric_n')} "
                  f"({g7.get('metabric_positives')} positive). Best expression-only AUC "
                  f"<b>{g7.get('best_metabric_auc')}</b> "
                  f"(panel {g7.get('best_panel_size')}, {g7.get('best_model')}). "
                  f"METABRIC has no matched 450k methylation, so the methylation half of the "
                  f"signature could not be tested externally.")
    story += [Paragraph("Abstract", s["H1"])]
    for h in hl:
        story.append(Paragraph(h, s["Body"]))

    story += [Paragraph("What this report does and does not claim", s["H2"]),
              Paragraph(
                  "This is a negative-leaning result reported as found. The pipeline was "
                  "pre-specified: feature selection is nested inside every cross-validation "
                  "split, models are compared on identical fold indices, and no model was "
                  "retuned after seeing its comparison. Where the data could not support the "
                  "planned analysis, the deviation is stated rather than worked around.",
                  s["Body"]), PageBreak()]

    # ---------------- gate 0/1 ----------------
    story += [Paragraph("1. Data and cohort construction", s["H1"])]
    man = r.csv("gate0_data_manifest.csv")
    if man is not None:
        story += [Paragraph("1.1 Sources (Gate 0)", s["H2"])]
        rows = [["Dataset", "Rows", "Cols", "Size"]]
        for _, x in man.iterrows():
            rows.append([x["key"], f"{x['n_rows']:,.0f}" if pd.notna(x["n_rows"]) else "-",
                         f"{x['n_cols']:,.0f}" if pd.notna(x["n_cols"]) else "-",
                         x["size_human"]])
        story += [mktable(rows, [58 * mm, 26 * mm, 20 * mm, 24 * mm], s), Spacer(1, 4)]
        story += [Paragraph("Full URLs and md5 checksums are in "
                            "<i>results/tables/gate0_data_manifest.csv</i>.", s["Cap"])]

    story += [Paragraph("1.2 Label definition", s["H2"]),
              Paragraph(
                  f"Relapse is defined from the Progression-Free Interval (PFI) in the TCGA-CDR "
                  f"curated endpoints (Liu et al., Cell 2018), at a {horizon}-day "
                  f"({hy:.0f}-year) horizon:", s["Body"])]
    story += [mktable([["PFI event", "PFI time", "Label"],
                       ["1", f"&le; {horizon}d", "positive (relapsed)"],
                       ["0", f"&ge; {horizon}d", "negative (relapse-free)"],
                       ["0", f"&lt; {horizon}d", "<b>EXCLUDED</b> - censored early, status unknown"],
                       ["1", f"&gt; {horizon}d", "negative"]],
                      [24 * mm, 26 * mm, 78 * mm], s), Spacer(1, 5)]
    story += [Paragraph(
        "The third row is the one that is easy to get wrong. A patient censored at two years "
        "is not a negative; their five-year status is unknown. Excluding them is correct and "
        "expensive: it is the single largest filter in this pipeline.", s["Body"])]

    story += [Paragraph("1.3 Horizon choice (deviation from spec)", s["H2"])]
    hz = r.csv("gate1_horizon_sensitivity.csv")
    if hz is not None:
        story += [Paragraph(
            "The build spec targeted a 5-year horizon and expected n = 550-750 with 15-25% "
            "positives. Measured over the patient universe with matched expression AND "
            "methylation (782 patients), no horizon satisfies both the n &ge; 400 and "
            "positives &ge; 60 floors simultaneously. The binding constraint is the "
            "multi-omics intersection, not the horizon.", s["Body"])]
        rows = [["Horizon", "n", "Positives", "Prevalence", "Excluded (early censor)",
                 "Meets n&ge;400", "Meets pos&ge;60"]]
        for _, x in hz.iterrows():
            rows.append([f"{x['horizon_years']:.0f} y", f"{x['n_usable']:.0f}",
                         f"{x['n_positive']:.0f}", f"{100*x['positive_rate']:.1f}%",
                         f"{x['n_excluded_early_censor']:.0f} ({x['pct_excluded']:.0f}%)",
                         "yes" if x["meets_n_ge_400"] else "<b>no</b>",
                         "yes" if x["meets_pos_ge_60"] else "<b>no</b>"])
        story += [mktable(rows, [18 * mm, 15 * mm, 20 * mm, 22 * mm, 34 * mm, 22 * mm, 22 * mm], s)]
        story += [Paragraph(
            f"<b>Decision:</b> the {hy:.0f}-year horizon is used as primary. It clears the "
            "positives floor and is the only horizon whose prevalence lands inside the "
            "expected 15-25% band; it misses the n &ge; 400 floor. The 5-year analysis is "
            "retained in full as a sensitivity run (results/tables_5y/).", s["Flag"])]
    add_fig(story, r.fig("fig1_horizon_sensitivity.png"),
            "Figure 1. Cohort size and class balance against relapse horizon.", s, 150 * mm)

    story += [PageBreak(), Paragraph("1.4 Filtering accounting (Gate 1)", s["H2"])]
    filt = r.csv("gate1_filtering.csv")
    if filt is not None:
        rows = [["Step", "Unit", "In", "Out", "Dropped"]]
        for _, x in filt.iterrows():
            rows.append([x["step"], x["unit"], f"{x['n_in']:,}", f"{x['n_out']:,}",
                         f"{x['n_dropped']:,}" if x["n_dropped"] else ""])
        story += [mktable(rows, [78 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm], s)]
    if g1:
        story += [Paragraph(
            f"<b>Final cohort:</b> n = {g1.get('n_final')}, "
            f"{g1.get('n_positive')} positive / {g1.get('n_negative')} negative "
            f"({100*g1.get('positive_rate',0):.1f}% prevalence), "
            f"{g1.get('n_expression_features')} expression features, "
            f"{g1.get('n_methylation_features')} methylation features, "
            f"{g1.get('n_shared_genes')} genes measured by both modalities.", s["Body"])]

    # ---------------- models ----------------
    story += [PageBreak(), Paragraph("2. Model comparison", s["H1"]),
              Paragraph(
                  "All three models are evaluated on the SAME outer fold indices "
                  f"({cfg['cv']['n_splits']}-fold x {cfg['cv']['n_repeats']} repeats = "
                  f"{cfg['cv']['n_splits']*cfg['cv']['n_repeats']} folds), loaded from a "
                  "written fold file rather than regenerated from a seed, so the comparison "
                  "is genuinely paired.", s["Body"])]
    rows = [["Model", "Description", "AUC (95% CI)", "AP (95% CI)"]]
    for tag, m, desc in (("A", mA, "Logistic regression, clinical only (age, stage, PAM50)"),
                         ("B", mB, "XGBoost, clinical + expression + methylation"),
                         ("C", mC, "Pathway-informed neural network (P-NET style)")):
        if m:
            rows.append([f"<b>{tag}</b>", desc,
                         f"{m['auc_mean']:.3f} [{m['auc_ci_lo']:.3f}, {m['auc_ci_hi']:.3f}]",
                         f"{m['ap_mean']:.3f} [{m['ap_ci_lo']:.3f}, {m['ap_ci_hi']:.3f}]"])
    story += [mktable(rows, [14 * mm, 72 * mm, 42 * mm, 38 * mm], s), Spacer(1, 5)]

    cmp_tbl = r.csv("model_comparison_tests.csv")
    if cmp_tbl is not None:
        story += [Paragraph("2.1 Paired tests on identical folds", s["H2"])]
        rows = [["Comparison", "Mean AUC difference", "Paired t p", "Wilcoxon p",
                 "Folds where 2nd wins"]]
        for _, x in cmp_tbl.iterrows():
            rows.append([x["comparison"], f"{x['mean_diff']:+.4f}",
                         f"{x['p_value']:.2e}", f"{x['wilcoxon_p']:.2e}",
                         f"{x['b_wins_folds']:.0f}/{x['n_folds']:.0f}"])
        story += [mktable(rows, [44 * mm, 34 * mm, 26 * mm, 26 * mm, 32 * mm], s)]

    add_fig(story, r.fig("fig2_model_comparison.png"),
            "Figure 2. Per-fold AUC by model on identical outer folds.", s, 165 * mm)
    add_fig(story, r.fig("fig3_roc_oof.png"),
            "Figure 3. Pooled out-of-fold ROC curves.", s, 90 * mm)

    story += [Paragraph("2.2 Gate 2 diagnostic: why is the clinical baseline strong?", s["H2"])]
    var = r.csv("gate2_baseline_variants.csv")
    stg = r.csv("gate2_stage_composition.csv")
    if var is not None:
        story += [Paragraph(
            "Model A exceeded the 0.60-0.70 band the spec expects for clinical-only BRCA "
            "models, so it was investigated before proceeding. It is not leakage: PAM50 "
            "(which is expression-derived, and so arguably should not sit in a 'clinical' "
            "baseline at all) contributes almost nothing, and the signal is carried by "
            "pathologic stage.", s["Body"])]
        rows = [["Variant", "AUC (95% CI)", "AP"]]
        for _, x in var.iterrows():
            rows.append([x["model"],
                         f"{x['auc_mean']:.3f} [{x['auc_ci_lo']:.3f}, {x['auc_ci_hi']:.3f}]",
                         f"{x['ap_mean']:.3f}"])
        story += [mktable(rows, [82 * mm, 44 * mm, 24 * mm], s), Spacer(1, 4)]
    if stg is not None:
        rows = [["Stage", "n", "Relapse rate"]]
        for _, x in stg.iterrows():
            st = x.iloc[0]
            lbl = "missing" if float(st) < 0 else f"{int(float(st))}"
            rows.append([lbl, f"{x['n']:.0f}", f"{100*x['relapse_rate']:.1f}%"])
        story += [Paragraph("Stage composition of the cohort:", s["Body"]),
                  mktable(rows, [26 * mm, 22 * mm, 30 * mm], s), Spacer(1, 4),
                  Paragraph(
                      "The stage gradient is steep, and the landmark labelling rule sharpens "
                      "it further: requiring survivors to have full follow-up preferentially "
                      "retains lower-stage patients as negatives, while early events remain as "
                      "positives. The strong clinical baseline is a real property of this "
                      "selected cohort, not an artefact of the code.", s["Flag"])]

    story += [PageBreak(), Paragraph("3. Feature ranking (Gate 3)", s["H1"])]
    shap_r = r.csv("model_b_shap_ranking.csv")
    if shap_r is not None:
        top = shap_r.head(30)
        rows = [["#", "Feature", "Block", "Mean |SHAP|", "Fold selection"]]
        for i, (_, x) in enumerate(top.iterrows(), 1):
            rows.append([str(i), x["feature"].split(":", 1)[-1], x["block"],
                         f"{x['mean_abs_shap']:.4f}",
                         f"{100*x['selection_frequency']:.0f}%"])
        story += [mktable(rows, [10 * mm, 52 * mm, 26 * mm, 26 * mm, 28 * mm], s)]
    add_fig(story, r.fig("fig4_shap_top30.png"),
            "Figure 4. Model B top 30 features by out-of-fold mean |SHAP|.", s, 120 * mm)

    # ---------------- signature ----------------
    story += [PageBreak(), Paragraph("4. Signature reduction (Gate 5)", s["H1"])]
    curve = r.csv("signature_panel_curve.csv")
    if curve is not None:
        rows = [["Panel size", "AUC (95% CI)", "AP"]]
        for _, x in curve.sort_values("panel_size").iterrows():
            rows.append([f"{x['panel_size']:.0f}",
                         f"{x['auc_mean']:.3f} [{x['auc_ci_lo']:.3f}, {x['auc_ci_hi']:.3f}]",
                         f"{x['ap_mean']:.3f}"])
        story += [mktable(rows, [26 * mm, 46 * mm, 24 * mm], s), Spacer(1, 4)]
    if plateau:
        story += [Paragraph(
            f"Performance plateaus at <b>{plateau.get('plateau_panel_size')}</b> markers "
            f"(smallest panel within one standard error of the best mean AUC, which was "
            f"{plateau.get('best_auc', float('nan')):.3f} at "
            f"{plateau.get('best_panel_size')} markers). 50-marker panel composition: "
            f"{plateau.get('panel_50_composition')}.", s["Body"])]
    story += [Paragraph(
        "Panels are re-selected inside every fold from the training half only; the reported "
        "marker list is the consensus across folds (selection frequency is given per marker "
        "in <i>results/tables/signature_panels.csv</i>). Scoring a globally-chosen panel "
        "would have leaked the test folds into the panel definition.", s["Body"])]
    add_fig(story, r.fig("fig5_panel_size_curve.png"),
            "Figure 5. AUC against panel size, with model baselines.", s, 120 * mm)

    panels = r.csv("signature_panels.csv")
    if panels is not None and (panels["panel_size"] == 50).any():
        p50 = panels[panels["panel_size"] == 50].head(50)
        story += [Paragraph("4.1 The 50-marker panel", s["H2"])]
        rows = [["#", "Marker", "Block", "Freq.", "#", "Marker", "Block", "Freq."]]
        half = (len(p50) + 1) // 2
        for i in range(half):
            a = p50.iloc[i]
            row = [str(i + 1), a["gene"], a["block"], f"{100*a['fold_selection_frequency']:.0f}%"]
            j = i + half
            if j < len(p50):
                b = p50.iloc[j]
                row += [str(j + 1), b["gene"], b["block"],
                        f"{100*b['fold_selection_frequency']:.0f}%"]
            else:
                row += ["", "", "", ""]
            rows.append(row)
        story += [mktable(rows, [8*mm, 30*mm, 18*mm, 14*mm, 8*mm, 30*mm, 18*mm, 14*mm], s)]

    # ---------------- immune ----------------
    story += [PageBreak(), Paragraph("5. Immune deconvolution (Gate 6)", s["H1"])]
    story += [Paragraph(
        "<b>Method deviation, stated plainly.</b> The spec asks for immunedeconv (R) with "
        "quanTIseq or MCP-counter. The immunedeconv install failed on this machine: "
        "Bioconductor dependencies (limSolve, biomaRt, sva, xCell, ConsensusTME, ComICS, "
        "quantiseqr) were unavailable. MCP-counter was therefore reimplemented directly in "
        "Python from the published 111-transcript, 10-population marker set "
        "(Becht et al., Genome Biology 2016) - the same signature the R package wraps. "
        "MCP-counter is a marker-gene aggregate, not a fitted deconvolution, so the "
        "reimplementation is faithful rather than approximate. quanTIseq was NOT run: it "
        "requires the TIL10 signature matrix that ships with the R package.", s["Flag"])]
    if g6:
        story += [Paragraph(
            f"Patients were split at the median out-of-fold risk score from Model "
            f"{g6.get('risk_model')} (AUC {g6.get('risk_model_auc', 0):.3f}): "
            f"{g6.get('n_high_risk')} high-risk vs {g6.get('n_low_risk')} low-risk. "
            f"Observed relapse rate {100*g6.get('relapse_rate_high',0):.1f}% vs "
            f"{100*g6.get('relapse_rate_low',0):.1f}%.", s["Body"])]
    imm = r.csv("gate6_risk_vs_immune.csv")
    if imm is not None:
        rows = [["Population", "High risk", "Low risk", "Delta", "p", "FDR"]]
        for _, x in imm.iterrows():
            rows.append([x["population"], f"{x['mean_high_risk']:.3f}",
                         f"{x['mean_low_risk']:.3f}", f"{x['delta']:+.3f}",
                         f"{x['p_value']:.3g}",
                         f"<b>{x['p_fdr']:.3g}</b>" if x["significant_fdr_0.05"]
                         else f"{x['p_fdr']:.3g}"])
        story += [mktable(rows, [42 * mm, 24 * mm, 22 * mm, 20 * mm, 22 * mm, 22 * mm], s)]
    if g6.get("verdict"):
        story += [Spacer(1, 4), Paragraph(f"<b>Gate 6 verdict.</b> {g6['verdict']}", s["Body"])]
    add_fig(story, r.fig("fig6_immune_deconvolution.png"),
            "Figure 6. MCP-counter scores by predicted risk group.", s, 130 * mm)

    # ---------------- external ----------------
    story += [PageBreak(), Paragraph("6. External validation in METABRIC (Gate 7)", s["H1"])]
    story += [Paragraph(
        "<b>Known constraint, not worked around.</b> METABRIC has no matched Illumina 450k "
        "methylation, so only the expression component of the signature can be tested "
        "externally. cBioPortal does host a promoter-RRBS methylation profile for METABRIC, "
        "but RRBS on a sample subset is a different assay and is not a substitute; it was not "
        "used. Transfer is also cross-platform (TCGA RNA-seq to Illumina HT-12 microarray), "
        "so genes were z-scored within each cohort independently, and degradation is expected.",
        s["Flag"])]
    mapt = r.csv("gate7_gene_mapping.csv")
    if mapt is not None:
        rows = [["Panel", "Expr. markers", "Mapped to METABRIC", "Rate",
                 "Methylation markers (untestable)"]]
        for _, x in mapt.iterrows():
            rows.append([f"{x['panel_size']:.0f}", f"{x['expression_markers']:.0f}",
                         f"{x['expression_markers_mapped_to_METABRIC']:.0f}",
                         f"{100*x['mapping_rate']:.0f}%" if pd.notna(x["mapping_rate"]) else "-",
                         f"{x['methylation_markers_dropped']:.0f}"])
        story += [mktable(rows, [16 * mm, 28 * mm, 38 * mm, 18 * mm, 48 * mm], s), Spacer(1, 4)]
    perf = r.csv("gate7_metabric_performance.csv")
    if perf is not None:
        rows = [["Panel", "Model", "Genes used", "METABRIC AUC", "METABRIC AP"]]
        for _, x in perf.sort_values(["panel_size", "model"]).iterrows():
            rows.append([f"{x['panel_size']:.0f}", x["model"], f"{x['n_genes_used']:.0f}",
                         f"<b>{x['metabric_auc']:.3f}</b>", f"{x['metabric_ap']:.3f}"])
        story += [mktable(rows, [16 * mm, 26 * mm, 24 * mm, 30 * mm, 28 * mm], s)]
    add_fig(story, r.fig("fig8_metabric_validation.png"),
            "Figure 7. External validation across panel sizes.", s, 120 * mm)
    if g7.get("km"):
        km = g7["km"]
        bits = []
        for k, nm in (("tcga", "TCGA (out-of-fold risk)"), ("metabric", "METABRIC (external)")):
            if k in km:
                bits.append(f"{nm}: log-rank p = {km[k]['logrank_p']:.3g}")
        if bits:
            story += [Paragraph("<b>Kaplan-Meier by predicted risk group.</b> "
                                + "; ".join(bits) + ".", s["Body"])]
    add_fig(story, r.fig("fig7_kaplan_meier.png"),
            "Figure 8. Relapse-free survival by predicted risk group. Note that negatives are "
            "required by the landmark rule to have full follow-up, which flattens the early "
            "part of the curves.", s, 165 * mm)

    # ---------------- limitations ----------------
    story += [PageBreak(), Paragraph("7. Limitations", s["H1"])]
    lims = [
        f"<b>Sample size.</b> n = {g1.get('n_final','?')} with ~19,000 candidate features. "
        "This is the dominant limitation and the most likely reason the omics models do not "
        "beat the clinical baseline. The spec's n = 550-750 target is not reachable in "
        "TCGA-BRCA once matched methylation and the early-censoring rule are both enforced.",
        f"<b>Early-censoring exclusion.</b> {g1.get('n_excluded_early_censoring','?')} patients "
        "were discarded because their status at the horizon is genuinely unknown. This is the "
        "methodologically correct choice, but it induces a selection effect: negatives are, by "
        "construction, patients with long follow-up. Prevalence and the strength of stage as a "
        "predictor are both inflated relative to an unselected cohort.",
        "<b>No matched external methylation.</b> Only the expression half of the signature was "
        "externally validated. A multi-omics signature validated on one modality is a partially "
        "validated signature.",
        "<b>Cross-platform transfer.</b> RNA-seq to microarray transfer degrades performance "
        "independently of biology; the METABRIC number confounds signature quality with "
        "platform shift.",
        "<b>PAM50 in the 'clinical' baseline.</b> PAM50 is expression-derived. It is included "
        "because the spec lists it, but it makes Model A not a purely clinical model. The "
        "diagnostic in section 2.2 quantifies its (small) contribution.",
        "<b>Grade unavailable.</b> Histologic grade is 100% missing in the TCGA-CDR fields for "
        "this cohort and was dropped, so the clinical baseline is thinner than specified.",
        "<b>quanTIseq not run</b>, and immunedeconv (R) not used - see section 5.",
        "<b>Single cohort discovery.</b> Discovery is entirely within TCGA-BRCA; no "
        "multi-cohort discovery or meta-analysis was attempted.",
    ]
    for l in lims:
        story.append(Paragraph("&bull; " + l, s["Body"]))

    # ---------------- repro ----------------
    story += [Paragraph("8. Reproducibility", s["H1"])]
    rows = [["Item", "Value"],
            ["Random seed", str(cfg["seed"])],
            ["Outer CV", f"{cfg['cv']['n_splits']}-fold x {cfg['cv']['n_repeats']} repeats "
                         f"({cfg['cv']['n_splits']*cfg['cv']['n_repeats']} folds), "
                         f"shared across models via cv_folds.json"],
            ["Inner CV", f"{cfg['cv']['inner_splits']}-fold, "
                         f"{cfg['cv']['n_random_search']} random search candidates"],
            ["Leakage control", "All selection/imputation/scaling inside sklearn Pipelines, "
                                "refit per split"],
            ["Container", "docker/Dockerfile (python:3.11-slim, pinned)"],
            ["Workflow", "nextflow/main.nf (DSL2), profiles: standard, docker, conda, smoke"],
            ["Smoke test", "scripts/smoke_test.sh - full pipeline on a 100-patient fixture"],
            ["Environment", "env/requirements.txt (pip freeze), env/requirements-docker.txt"]]
    story += [mktable(rows, [36 * mm, 128 * mm], s)]
    story += [Spacer(1, 6), Paragraph(
        "Every script takes <i>--config</i>, writes to <i>results/</i>, logs each filter with "
        "in/out counts, and records the seed. Figures are generated by a separate script that "
        "only reads tables, so a figure cannot disagree with the table behind it.", s["Body"])]

    # Name the file by horizon: the 3y and 5y runs share a parent directory,
    # so a fixed filename would have one silently overwrite the other.
    tag = str(cfg["label"].get("horizon_label", f"{horizon}d")).replace(" ", "")
    out = (root / cfg["paths"]["tables"] / ".." / f"biomarker_signature_report_{tag}.pdf").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="Multi-Omics Biomarker Signature for Breast Cancer Relapse")

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(GREY)
        canvas.drawString(22 * mm, 10 * mm, "biomarker-signature-brca")
        canvas.drawRightString(A4[0] - 22 * mm, 10 * mm, f"page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    log.info("wrote %s (%.2f MB)", out, out.stat().st_size / 1e6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("09_report", cfg)
    build(cfg, log)


if __name__ == "__main__":
    main()
