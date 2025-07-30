# -*- coding: utf-8 -*-
from __future__ import annotations

import io
import os
import re
import zipfile
import unicodedata
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
from streamlit_image_coordinates import streamlit_image_coordinates  # clic pour positionner


# =========================
# Helpers
# =========================

def slugify(text: str, allow_unicode: bool = False) -> str:
    if allow_unicode:
        text = unicodedata.normalize("NFKC", text)
    else:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def normalize_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        return s.strip().lower()

    cols = {norm(c): c for c in df.columns}
    prenom = cols.get("prenom") or cols.get("prénom") or cols.get("first") or cols.get("firstname")
    nom = cols.get("nom") or cols.get("last") or cols.get("lastname")
    ent = cols.get("entreprise") or cols.get("societe") or cols.get("company") or cols.get("organisation") or cols.get("organization")
    return prenom, nom, ent


def pil_font_from_upload(uploaded_font, size_px: int) -> ImageFont.FreeTypeFont:
    if uploaded_font is None:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size_px)
        except Exception:
            return ImageFont.load_default()
    else:
        return ImageFont.truetype(io.BytesIO(uploaded_font.getbuffer()), size_px)


def rgba_from_hex(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def pymupdf_rgb_from_hex(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def render_template_preview(template_bytes: bytes, dpi: int = 200) -> Tuple[Image.Image, Tuple[float, float], float]:
    """Rend l’image d’aperçu à partir du PDF."""
    doc = fitz.open(stream=template_bytes, filetype="pdf")
    page = doc[0]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.info["dpi"] = (dpi, dpi)
    page_size_pt = (page.rect.width, page.rect.height)
    doc.close()
    return img, page_size_pt, zoom


def draw_preview(img: Image.Image, cfg: Dict[str, Any], row: Dict[str, str], font_upload, preview_zoom: float) -> Image.Image:
    """Dessine les champs sur une copie de l’image d’aperçu."""
    preview = img.copy()
    W, H = preview.size
    draw = ImageDraw.Draw(preview)

    def draw_field(text: str, pos_pct: Tuple[float, float], size_pt: float, color_hex: str,
                   upper: bool = False, anchor_mode: str = "left"):
        if not text:
            return
        if upper:
            text = text.upper()
        size_px = max(8, int(size_pt * preview_zoom))  # 1pt -> px via zoom
        font = pil_font_from_upload(font_upload, size_px)
        x = int(pos_pct[0] / 100.0 * W)
        y = int(pos_pct[1] / 100.0 * H)

        # Décalage horizontal si ancrage = centre (mesure via textbbox)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w_text = bbox[2] - bbox[0]
        except Exception:
            w_text = 0
        if anchor_mode == "center":
            x = x - w_text // 2

        try:
            draw.text((x, y), text, font=font, fill=rgba_from_hex(color_hex))
        except TypeError:
            draw.text((x, y), text, font=font)

    prenom = str(row.get("prenom", "")).strip()
    nom = str(row.get("nom", "")).strip()
    entreprise = str(row.get("entreprise", "")).strip()

    draw_field(prenom, cfg["prenom_pos"], cfg["prenom_size"], cfg["prenom_color"], cfg["prenom_upper"], cfg["prenom_anchor"])
    draw_field(nom, cfg["nom_pos"], cfg["nom_size"], cfg["nom_color"], cfg["nom_upper"], cfg["nom_anchor"])
    draw_field(entreprise, cfg["ent_pos"], cfg["ent_size"], cfg["ent_color"], cfg["ent_upper"], cfg["ent_anchor"])

    return preview


def generate_cards_pdf(
    template_bytes: bytes,
    data_rows: List[Dict[str, str]],
    cfg: Dict[str, Any],
    font_upload,
    out_mode: str = "separate",  # "separate" or "merged"
    filename_pattern: str = "{prenom}_{nom}.pdf",
    write_server_dir: Optional[str] = None,
) -> io.BytesIO:
    """Écrit les cartes PDF et retourne un ZIP en mémoire."""
    tmp_font_path: Optional[str] = None
    fontname = "helv"
    fontfile = None
    if font_upload is not None:
        tmp_font_path = os.path.join(".", f"_font_{int(datetime.utcnow().timestamp())}.ttf")
        with open(tmp_font_path, "wb") as f:
            f.write(font_upload.getbuffer())
        fontfile = tmp_font_path

    # Métriques pour largeur en points
    font_metric = fitz.Font(fontfile=fontfile) if fontfile else None

    src = fitz.open(stream=template_bytes, filetype="pdf")
    page0 = src[0]
    page_w, page_h = page0.rect.width, page0.rect.height

    zip_buf = io.BytesIO()
    zf = zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED)
    if write_server_dir:
        os.makedirs(write_server_dir, exist_ok=True)

    def text_width_pt(text: str, fontsize: float) -> float:
        if font_metric is not None:
            return float(font_metric.text_length(text, fontsize))
        return float(fitz.get_text_length(text, fontname=fontname, fontsize=fontsize))

    def place_text(page: fitz.Page, text: str, pos_pct: Tuple[float, float], size_pt: float,
                   color_hex: str, upper: bool, anchor_mode: str):
        if not text:
            return
        if upper:
            text = text.strip().upper()
        x = pos_pct[0] / 100.0 * page_w
        y = pos_pct[1] / 100.0 * page_h
        if anchor_mode == "center":
            x -= text_width_pt(text, float(size_pt)) / 2.0
        color = pymupdf_rgb_from_hex(color_hex)
        page.insert_text((x, y), text, fontsize=float(size_pt), fontname=fontname, fontfile=fontfile, color=color)

    def write_one(page: fitz.Page, row: Dict[str, str]):
        place_text(page, row.get("prenom", ""), cfg["prenom_pos"], cfg["prenom_size"], cfg["prenom_color"], cfg["prenom_upper"], cfg["prenom_anchor"])
        place_text(page, row.get("nom", ""), cfg["nom_pos"], cfg["nom_size"], cfg["nom_color"], cfg["nom_upper"], cfg["nom_anchor"])
        place_text(page, row.get("entreprise", ""), cfg["ent_pos"], cfg["ent_size"], cfg["ent_color"], cfg["ent_upper"], cfg["ent_anchor"])

    if out_mode == "merged":
        out_doc = fitz.open()
        for row in data_rows:
            out_doc.insert_pdf(src, from_page=0, to_page=0)
            write_one(out_doc[-1], row)
        merged_name = "cartes_mergees.pdf"
        out_bytes = out_doc.tobytes()
        zf.writestr(merged_name, out_bytes)
        if write_server_dir:
            with open(os.path.join(write_server_dir, merged_name), "wb") as f:
                f.write(out_bytes)
        out_doc.close()
    else:
        for row in data_rows:
            out_doc = fitz.open()
            out_doc.insert_pdf(src, from_page=0, to_page=0)
            write_one(out_doc[0], row)
            base = filename_pattern.format(
                prenom=slugify(str(row.get("prenom", "")) or "prenom"),
                nom=slugify(str(row.get("nom", "")) or "nom"),
                entreprise=slugify(str(row.get("entreprise", "")) or "entreprise"),
            )
            if not base.lower().endswith(".pdf"):
                base += ".pdf"
            out_bytes = out_doc.tobytes()
            zf.writestr(base, out_bytes)
            if write_server_dir:
                with open(os.path.join(write_server_dir, base), "wb") as f:
                    f.write(out_bytes)
            out_doc.close()

    zf.close()
    zip_buf.seek(0)

    if fontfile:
        try:
            os.remove(tmp_font_path)  # type: ignore[arg-type]
        except Exception:
            pass
    src.close()
    return zip_buf


# =========================
# UI Streamlit
# =========================

st.set_page_config(page_title="Générateur de cartes personnel (PDF)", layout="wide")
st.title("🪪 Générateur de cartes de personnel (PDF)")

st.markdown(
    "Chargez un **PDF modèle** (ex. Canva, 1 page) et un **CSV** (`Prénom, Nom, Entreprise`). "
    "Réglez le placement & style puis générez toutes les cartes."
)

with st.sidebar:
    st.header("1) Import")
    pdf_file = st.file_uploader("Modèle PDF (1 page)", type=["pdf"])
    csv_file = st.file_uploader("CSV (Prénom, Nom, Entreprise)", type=["csv"])
    font_upload = st.file_uploader("Police .ttf (optionnel)", type=["ttf"])

    st.divider()
    st.header("Sortie")
    out_mode_label = st.radio(
        "Format de sortie",
        ["Un PDF par personne", "Un seul PDF multi-pages"],
        index=0,
        key="out_mode"  # <-- clé unique
    )
    out_mode_key = "separate" if out_mode_label.startswith("Un PDF par personne") else "merged"
    filename_pattern = st.text_input("Patron nom de fichier (si 1/pdf)", "{prenom}_{nom}.pdf",
                                     help="{prenom} {nom} {entreprise} disponibles")
    server_dir = st.text_input("Dossier serveur (optionnel, exécution locale)", value="")
    dpi_preview = st.slider("DPI de prévisualisation", 120, 300, 200, 20)

# Lecture CSV
df: Optional[pd.DataFrame] = None
if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        try:
            df = pd.read_csv(csv_file, sep=";")
        except Exception as e:
            st.error(f"Impossible de lire le CSV : {e}")

if df is not None:
    st.subheader("Aperçu CSV")
    st.dataframe(df.head(10), use_container_width=True)

# Mapping colonnes
if df is not None:
    c1, c2, c3 = st.columns(3)
    auto_prenom, auto_nom, auto_ent = normalize_columns(df)

    def default_index(col_name: Optional[str]) -> int:
        try:
            return df.columns.get_loc(col_name) if col_name in df.columns else 0  # type: ignore[operator]
        except Exception:
            return 0

    with c1:
        col_prenom = st.selectbox("Colonne Prénom", options=df.columns.tolist(), index=default_index(auto_prenom))
    with c2:
        col_nom = st.selectbox("Colonne Nom", options=df.columns.tolist(), index=default_index(auto_nom))
    with c3:
        col_ent = st.selectbox("Colonne Entreprise", options=df.columns.tolist(), index=default_index(auto_ent))

# ------ Config design par défaut + état persistant ------
default_cfg = {
    "prenom_pos": (20.0, 40.0),  # (x%, y%)
    "nom_pos": (20.0, 50.0),
    "ent_pos": (20.0, 60.0),
    "prenom_size": 18.0,
    "nom_size": 18.0,
    "ent_size": 14.0,
    "prenom_color": "#000000",
    "nom_color": "#000000",
    "ent_color": "#000000",
    "prenom_upper": False,
    "nom_upper": True,
    "ent_upper": False,
    # Ancrage: "left" ou "center"
    "prenom_anchor": "left",
    "nom_anchor": "left",
    "ent_anchor": "left",
}
if "cfg" not in st.session_state:
    st.session_state.cfg = default_cfg.copy()
cfg = st.session_state.cfg

template_bytes = pdf_file.getvalue() if pdf_file is not None else None

# ====== 2) Aperçu & positionnement par clic ======
st.subheader("2) Aperçu et positionnement par clic")

try:
    col_preview, col_style = st.columns([3, 2], vertical_alignment="top")
except TypeError:
    col_preview, col_style = st.columns([3, 2])

with col_style:
    st.markdown("### Champ à positionner")
    target = st.radio(
        "Choisissez puis cliquez sur l’aperçu à gauche :",
        ["Prénom", "Nom", "Entreprise"],
        horizontal=True,
        key="target_field"  # <-- clé unique
    )

    st.markdown("### Style & ancrage")
    tabs = st.tabs(["Prénom", "Nom", "Entreprise"])
    # PRÉNOM
    with tabs[0]:
        cfg["prenom_size"]  = st.slider("Taille Prénom (pt)", 8.0, 72.0, float(cfg["prenom_size"]), 0.5)
        cfg["prenom_color"] = st.color_picker("Couleur Prénom", cfg["prenom_color"])
        cfg["prenom_upper"] = st.checkbox("Prénom en MAJUSCULES", value=bool(cfg["prenom_upper"]))
        anch_prenom = st.radio(
            "Ancrage (point cliqué)", ["Gauche", "Centre"],
            index=0 if cfg["prenom_anchor"] == "left" else 1,
            horizontal=True,
            key="anchor_prenom"  # <-- clé unique
        )
        cfg["prenom_anchor"] = "left" if anch_prenom == "Gauche" else "center"
    # NOM
    with tabs[1]:
        cfg["nom_size"]  = st.slider("Taille Nom (pt)", 8.0, 72.0, float(cfg["nom_size"]), 0.5)
        cfg["nom_color"] = st.color_picker("Couleur Nom", cfg["nom_color"])
        cfg["nom_upper"] = st.checkbox("Nom en MAJUSCULES", value=bool(cfg["nom_upper"]))
        anch_nom = st.radio(
            "Ancrage (point cliqué)", ["Gauche", "Centre"],
            index=0 if cfg["nom_anchor"] == "left" else 1,
            horizontal=True,
            key="anchor_nom"  # <-- clé unique
        )
        cfg["nom_anchor"] = "left" if anch_nom == "Gauche" else "center"
    # ENTREPRISE
    with tabs[2]:
        cfg["ent_size"]  = st.slider("Taille Entreprise (pt)", 8.0, 72.0, float(cfg["ent_size"]), 0.5)
        cfg["ent_color"] = st.color_picker("Couleur Entreprise", cfg["ent_color"])
        cfg["ent_upper"] = st.checkbox("Entreprise en MAJUSCULES", value=bool(cfg["ent_upper"]))
        anch_ent = st.radio(
            "Ancrage (point cliqué)", ["Gauche", "Centre"],
            index=0 if cfg["ent_anchor"] == "left" else 1,
            horizontal=True,
            key="anchor_ent"  # <-- clé unique
        )
        cfg["ent_anchor"] = "left" if anch_ent == "Gauche" else "center"

    st.session_state.cfg = cfg  # persiste

with col_preview:
    if template_bytes is None:
        st.info("Chargez un modèle PDF pour afficher l’aperçu.")
    else:
        preview_img, page_size_pt, preview_zoom = render_template_preview(template_bytes, dpi=dpi_preview)
        max_w = 900
        canvas_w = min(max_w, preview_img.width)
        canvas_h = int(canvas_w * preview_img.height / preview_img.width)

        example = {"prenom": "Camille", "nom": "Durand", "entreprise": "Acme SA"}
        if df is not None and len(df) > 0:
            example = {"prenom": df.iloc[0][col_prenom], "nom": df.iloc[0][col_nom], "entreprise": df.iloc[0][col_ent]}

        bg = draw_preview(preview_img, cfg, example, font_upload, preview_zoom).resize((canvas_w, canvas_h))
        click = streamlit_image_coordinates(bg, key="coord_preview")
        if click is not None and all(k in click for k in ("x", "y", "width", "height")):
            x_pct = round(click["x"] / click["width"] * 100.0, 2)
            y_pct = round(click["y"] / click["height"] * 100.0, 2)
            if target == "Prénom":
                cfg["prenom_pos"] = (x_pct, y_pct)
            elif target == "Nom":
                cfg["nom_pos"] = (x_pct, y_pct)
            else:
                cfg["ent_pos"] = (x_pct, y_pct)
            st.session_state.cfg = cfg

        st.caption("Cliquez pour positionner le champ sélectionné. Choisissez l’**ancrage** (gauche/centre) dans les onglets à droite.")

# ====== 4) Génération ======
st.subheader("4) Génération")
can_generate = (template_bytes is not None) and (df is not None) and all(
    c in df.columns for c in [col_prenom, col_nom, col_ent]) if df is not None else False

if not can_generate:
    st.warning("Chargez un PDF, un CSV et vérifiez le mapping des colonnes.")
else:
    if st.button("⚙️ Générer toutes les cartes"):
        with st.spinner("Génération des PDF en cours..."):
            rows: List[Dict[str, str]] = []
            assert df is not None
            for _, r in df.iterrows():
                rows.append({
                    "prenom": "" if pd.isna(r[col_prenom]) else str(r[col_prenom]),
                    "nom": "" if pd.isna(r[col_nom]) else str(r[col_nom]),
                    "entreprise": "" if pd.isna(r[col_ent]) else str(r[col_ent]),
                })

            cfg = st.session_state.get("cfg", default_cfg)
            write_dir = server_dir.strip() or None

            try:
                zip_buf = generate_cards_pdf(
                    template_bytes=template_bytes,
                    data_rows=rows,
                    cfg=cfg,
                    font_upload=font_upload,
                    out_mode="merged" if out_mode_key == "merged" else "separate",
                    filename_pattern=filename_pattern,
                    write_server_dir=write_dir,
                )
            except Exception as e:
                st.error(f"Erreur pendant la génération : {e}")
                st.stop()

            zip_name = f"cartes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            st.success("Terminé ✅")
            st.download_button("⬇️ Télécharger le ZIP", data=zip_buf, file_name=zip_name, mime="application/zip")

            if write_dir:
                st.info(f"Fichiers également écrits côté serveur dans : `{write_dir}`.\n"
                        "En déploiement en ligne, utilisez le ZIP à télécharger.")
