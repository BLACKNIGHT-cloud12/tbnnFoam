"""Build + verify the Zenodo preprint deposit package for the tbnnFoam manuscript.
Derives all metadata from docs/latex/main.tex (no hand-typed drift).
Run with Windows python from repo root.
"""
import hashlib, json, os, re, shutil, zipfile

REPO = os.getcwd()
SRC_PDF = os.path.join("docs", "tbnnFoam_preprint.pdf")
SRC_TEX = os.path.join("docs", "latex", "main.tex")
SRC_BBL = os.path.join("docs", "latex", "main.bbl")
SRC_BIB = os.path.join("docs", "latex", "references.bib")
SRC_FIG = os.path.join("docs", "latex", "figures")
PKG = os.path.join("zenodo_preprint_package")
SOFTWARE_DOI = "10.5281/zenodo.23799524"
GITHUB = "https://github.com/BLACKNIGHT-cloud12/tbnnFoam"
AFFIL = "Department of Civil Engineering, Jadavpur University, India"
PUB_DATE = "2026-11-01"  # actual deposit date
VERSION = "1.0"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _common_convert(s):
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    for cmd in ("\\overline", "\\mathrm"):
        s = s.replace(cmd, "")
    s = re.sub(r"\\,", " ", s)
    s = s.replace(r"\%", "%")
    s = s.replace(r"\min", "min").replace(r"\max", "max")
    rep = {r"\propto": "∝", r"\approx": "≈", r"\ge": "≥", r"\le": "≤",
           r"\times": "×", r"\sigma": "σ", r"\tau": "τ", r"\omega": "ω",
           r"\kappa": "κ", r"\nu": "ν"}
    for k, v in rep.items():
        s = s.replace(k, v)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("$", "")
    s = re.sub(r"[{}]", "", s)
    s = s.replace("---", "—").replace("--", "–")
    s = s.replace("~", " ")
    return s

def latex_to_text(s):
    """LaTeX abstract -> plain text (for README). Paragraph- and list-preserving."""
    s = re.sub(r"\\begin\{enumerate\}", "", s)
    s = re.sub(r"\\end\{enumerate\}", "", s)
    s = re.sub(r"\\item\s*", "• ", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", s)
    s = _common_convert(s)
    paras = [re.sub(r"[ \t]+", " ", p).strip() for p in re.split(r"\n\s*\n", s)]
    return "\n\n".join(p for p in paras if p)

def latex_to_html(s):
    """LaTeX abstract -> Zenodo-safe HTML (allowed tags only)."""
    s = re.sub(r"\\begin\{enumerate\}", "<ul>", s)
    s = re.sub(r"\\end\{enumerate\}", "</ul>", s)
    s = re.sub(r"\\item\s*", "<li>", s)
    s = re.sub(r"\\textbf\{([^{}]*)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\texttt\{([^{}]*)\}", r"<code>\1</code>", s)
    s = _common_convert(s)
    # NOTE: <sup> is NOT in Zenodo's description HTML whitelist — keep exponents plain.
    s = re.sub(r"<li>(.*?)(?=<li>|</ul>)", r"<li>\1</li>", s, flags=re.S)
    s = re.sub(r"</li>\s*<li>", "</li><li>", s)
    s = re.sub(r"\n\s*\n", "</p><p>", s)
    s = "<p>" + re.sub(r"\s+", " ", s).replace("</p><p>", "</p>\n<p>").strip() + "</p>"
    s = s.replace(" <ul>", "</p><ul>").replace("</ul></p>", "</ul><p>")
    s = re.sub(r"<p>\s*<p>", "<p>", s)
    s = re.sub(r"</p>\s*</p>", "</p>", s)
    s = re.sub(r"<p>\s*(<ul>)", r"\1", s)
    s = re.sub(r"(</ul>)\s*</p>", r"\1", s)
    return s

# ---------- read manuscript ----------
tex = open(SRC_TEX, encoding="utf-8").read()
abstract_latex = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1).strip()
title = re.search(r"\\title\{\\textbf\{([^}]*)\}\}", tex).group(1).strip()

# ---------- bibliography: parse + format (CITED only) ----------
bib = open(SRC_BIB, encoding="utf-8").read()
cited_keys = set()
for grp in re.findall(r"\\cite\{([^}]*)\}", tex):
    cited_keys.update(k.strip() for k in grp.split(","))

def clean_value(v):
    v = re.sub(r"[{}]", "", v)
    umap = {'\\"U': "Ü", '\\"u': "ü", '\\"o': "ö", '\\"a': "ä", r"\ss": "ß", r"\&": "&",
            "\\'a": "á", "\\'A": "Á", "\\'e": "é", "\\'E": "É", "\\'i": "í",
            "\\'o": "ó", "\\'u": "ú"}
    for k, r in umap.items():
        v = v.replace(k, r)
    v = v.replace("--", "–")
    return re.sub(r"\s+", " ", v).strip()

def parse_fields(body):
    fields = {}
    for line in body.splitlines():
        mm = re.match(r'\s*(\w+)\s*=\s*([{"]).*?\2\s*,?\s*$', line)
        if not mm:
            mm = re.match(r'\s*(\w+)\s*=\s*(\{.*\})\s*,?\s*$', line)
            if mm:
                fields[mm.group(1).lower()] = clean_value(mm.group(2)[1:-1])
                continue
            mm = re.match(r'\s*(\w+)\s*=\s*"(.*)"\s*,?\s*$', line)
            if mm:
                fields[mm.group(1).lower()] = clean_value(mm.group(2))
            continue
        delim = mm.group(2)
        val = line[mm.start(2) + 1: line.rfind(delim)]
        fields[mm.group(1).lower()] = clean_value(val)
    return fields

def fmt_ref(fields):
    authors = re.sub(r"\s+", " ", fields.get("author", "")).replace(" and ", "; ").rstrip(".")
    year = fields.get("year", "")
    title = fields.get("title", "")
    if fields.get("journal"):
        vol, num = fields.get("volume", ""), fields.get("number", "")
        loc = f"{vol}({num})" if num else vol
        pages = fields.get("pages", "")
        tail = ", ".join(x for x in [fields["journal"], loc, pages] if x)
        return f"{authors} ({year}). {title}. {tail}."
    if fields.get("booktitle"):
        pages = fields.get("pages", "")
        return f"{authors} ({year}). {title}. In {fields['booktitle']}" + (f", {pages}." if pages else ".")
    if fields.get("publisher"):
        addr = fields.get("address", "")
        return f"{authors} ({year}). {title}. {fields['publisher']}" + (f", {addr}." if addr else ".")
    if fields.get("institution"):
        return f"{authors} ({year}). {title}. {fields['institution']}."
    return f"{authors} ({year}). {title}."

entries = []
for mm in re.finditer(r"@(\w+)\{([^,]+),\n(.*?)\n\}", bib, re.S):
    key = mm.group(2).strip()
    if key in cited_keys:
        entries.append(fmt_ref(parse_fields(mm.group(3))))
assert len(entries) == 14, f"expected 14 cited references, got {len(entries)}"

# ---------- package dir ----------
if os.path.isdir(PKG):
    shutil.rmtree(PKG)
os.makedirs(PKG)
shutil.copy2(SRC_PDF, os.path.join(PKG, "final_preprint.pdf"))

src_zip_path = os.path.join(PKG, "final_preprint_latex_source.zip")
with zipfile.ZipFile(src_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(SRC_TEX, "latex_source/main.tex")
    z.write(SRC_BIB, "latex_source/references.bib")
    z.write(SRC_BBL, "latex_source/main.bbl")
    for f in sorted(os.listdir(SRC_FIG)):
        z.write(os.path.join(SRC_FIG, f), "latex_source/figures/" + f)

# ---------- metadata JSON (flat Zenodo schema) ----------
metadata = {
    "title": title,
    "upload_type": "publication",
    "publication_type": "preprint",
    "publication_date": PUB_DATE,
    "creators": [{"name": "Ghosh, Soumik", "affiliation": AFFIL}],
    "description": latex_to_html(abstract_latex),
    "access_right": "open",
    "license": "cc-by-4.0",
    "version": VERSION,
    "keywords": ["Turbulence Modeling", "Machine Learning", "Tensor Basis Neural Network",
                 "OpenFOAM", "Secondary Flow of the Second Kind", "Computational Fluid Dynamics"],
    "related_identifiers": [
        {"identifier": SOFTWARE_DOI, "relation": "isSupplementedBy", "resource_type": "software"},
        {"identifier": GITHUB, "relation": "isSupplementedBy", "resource_type": "software"},
    ],
    "references": entries,
    "notes": ("This record is the research preprint. The companion software record "
              "(DOI: " + SOFTWARE_DOI + ") archives the tbnnFoam v1.0.0 solver release under an MIT license. "
              "The third-party benchmark DNS data is not redistributed with either record; "
              "the acquisition and conversion workflow is documented in the repository "
              "(data/benchmark_dns/README.md)."),
}
json_path = os.path.join(PKG, "zenodo_preprint.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

# ---------- README ----------
abstract_plain = latex_to_text(abstract_latex)
readme = f"""# {title}

**Author:** Soumik Ghosh ({AFFIL})

**Version:** {VERSION} · **Deposit date:** {PUB_DATE} · **License:** CC BY 4.0 (paper) — the companion software is MIT-licensed

## Abstract

{abstract_plain}

## Related software

This preprint is the research paper accompanying **tbnnFoam v1.0.0**, archived separately on Zenodo:
DOI [{SOFTWARE_DOI}](https://doi.org/{SOFTWARE_DOI}) (MIT license). Source code: {GITHUB}

The third-party benchmark DNS data is not redistributed with this deposit; the acquisition and
conversion workflow is documented in the software repository (`data/benchmark_dns/README.md`).

## How to cite

> Ghosh, S. ({PUB_DATE[:4]}). *{title}*. Zenodo. Preprint, version {VERSION}.
> The DOI for this record is assigned by Zenodo when the record is published and is displayed on the record page.
> Please also cite the companion software record ({SOFTWARE_DOI}) when the solver itself is used.

## Files in this deposit

All six files in this folder are intended for upload to the Zenodo record:

- `final_preprint.pdf` — the manuscript (28 pages).
- `README.md` — this file.
- `final_preprint_latex_source.zip` — LaTeX source (main.tex, references.bib, main.bbl, figures) for reproducible compilation.
- `zenodo_preprint.json` — prepared Zenodo metadata (machine-readable reference for the record).
- `final_preprint.pdf.sha256`, `final_preprint_latex_source.zip.sha256` — integrity checksums.
"""
with open(os.path.join(PKG, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme)

# ---------- checksums ----------
pdf_hash = sha256(os.path.join(PKG, "final_preprint.pdf"))
src_hash = sha256(src_zip_path)
with open(os.path.join(PKG, "final_preprint.pdf.sha256"), "w") as f:
    f.write(f"{pdf_hash}  final_preprint.pdf\n")
with open(os.path.join(PKG, "final_preprint_latex_source.zip.sha256"), "w") as f:
    f.write(f"{src_hash}  final_preprint_latex_source.zip\n")

# ---------- outer archival zip ----------
outer = "tbnnFoam_preprint_zenodo_package_v1.0.zip"
with zipfile.ZipFile(outer, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(PKG):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, "."))

# ================= VERIFICATION BATTERY =================
print("=" * 30, "VERIFICATION", "=" * 30)
import fitz
doc = fitz.open(os.path.join(PKG, "final_preprint.pdf"))
print(f"[PDF] pages={doc.page_count}")
def norm(s):
    s = re.sub(r"-\n", "", s)
    return re.sub(r"\s+", " ", s)
p1 = norm(doc[0].get_text())
for needle in ["Soumik Ghosh", "Jadavpur University",
               "In-Situ Coupling of Tensor Basis Neural Networks for Turbulence Modeling in OpenFOAM"]:
    print(f"[PDF] titlepage has '{needle[:50]}…':", needle in p1)
full_text = norm("".join(doc[i].get_text() for i in range(doc.page_count)))
print("[PDF] availability fix present:", "documented workflow for obtaining the third-party DNS benchmark data" in full_text)
print("[PDF] Pirozzoli pages 631–655:", "631–655" in full_text, "| old 634–655 absent:", "634–655" not in full_text)
print("[PDF] Banerjee comma removed:", "invariants, versus" not in full_text and "invariants versus" in full_text)
print("[PDF] Prandtl 'Über … Turbulenz':", "Über die ausgebildete Turbulenz" in full_text)
print("[PDF] page-26 duplicate paragraph removed:", "available in the public repository:" not in full_text)
print("[PDF] CI three-environment present:", "17 passed, 4 skipped there" in full_text)
fonts = set()
for i in range(doc.page_count):
    for fnt in doc.get_page_fonts(i):
        fonts.add(fnt[3])
print(f"[PDF] fonts={len(fonts)}, all embedded subsets (contain '+'):", all("+" in f for f in fonts))
cites = set()
for grp in re.findall(r"\\cite\{([^}]*)\}", tex):
    cites.update(k.strip() for k in grp.split(","))
bbl = open(SRC_BBL, encoding="utf-8").read()
bibitems = set(re.findall(r"\\bibitem\{([^}]*)\}", bbl))
print(f"[REFS] cite keys={len(cites)}, bibitems={len(bibitems)}, missing-in-bbl={sorted(cites - bibitems)}, uncited={sorted(bibitems - cites)}")
figs_ref = set(re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]*)\}", tex))
figs_dir = set(os.listdir(SRC_FIG))
print(f"[FIGS] referenced={len(figs_ref)}, on disk={len(figs_dir)}, match={figs_ref == figs_dir}")
meta = json.load(open(json_path, encoding="utf-8"))
print("[META] flat schema: upload_type =", meta["upload_type"], "| publication_type =", meta["publication_type"])
print("[META] license is plain string:", meta["license"] == "cc-by-4.0", "→", repr(meta["license"]))
print("[META] no <sup> in description:", "<sup>" not in meta["description"],
      "| <code> whitelisted and used:", "<code>" in meta["description"])
print("[META] title exact match:", meta["title"] == title)
print("[META] description has tbnnFoam:", "tbnnFoam" in meta["description"],
      "| no leftover backslash:", "\\" not in meta["description"])
li_open, li_close = meta["description"].count("<li>"), meta["description"].count("</li>")
p_open, p_close = meta["description"].count("<p>"), meta["description"].count("</p>")
print("[META] <li> balanced:", li_open == li_close, f"({li_open}/{li_close})",
      "| <p> balanced:", p_open == p_close, f"({p_open}/{p_close})")
print("[META] references count:", len(meta["references"]), "(must equal PDF bibliography = 14)")
weller = [r for r in meta["references"] if "Weller" in r or "tensorial approach" in r.lower()]
print("[META] Weller excluded:", not weller)
print("[META] Banerjee comma removed:", not any("invariants, versus" in r for r in meta["references"]))
print("[META] Pirozzoli 631–655:", any("631–655" in r for r in meta["references"]))
print("[META] Prandtl clean:", any("Über die ausgebildete Turbulenz" in r and '\\"' not in r for r in meta["references"]))
print("[META] related software DOI:", meta["related_identifiers"][0]["identifier"])
print("[SHA256] pdf:", pdf_hash)
print("[SHA256] src zip:", src_hash)
tmp = os.path.join(PKG, "_extract_test")
with zipfile.ZipFile(outer) as z:
    z.extractall(tmp)
re_extracted = os.path.join(tmp, "zenodo_preprint_package", "final_preprint.pdf")
print("[ZIP] extract ok, inner pdf sha matches:", sha256(re_extracted) == pdf_hash)
shutil.rmtree(tmp)
banned = []
for root, _, files in os.walk(PKG):
    for fn in files:
        if fn.endswith(".dat") or "raw_dns" in fn.lower() or fn.startswith("tbnnFoam-1.0.0-source.zip"):
            banned.append(fn)
print("[SCOPE] banned files present:", banned or "none")
print("=" * 30, "FORMATTED REFERENCES (all 14)", "=" * 30)
for i, r in enumerate(meta["references"], 1):
    print(f"[{i}] {r}")
print("=" * 30, "INVENTORY", "=" * 30)
for root, _, files in os.walk(PKG):
    for fn in sorted(files):
        p = os.path.join(root, fn)
        print(f"{os.path.getsize(p):>10,}  {os.path.relpath(p)}")
print(f"{os.path.getsize(outer):>10,}  {outer}")
