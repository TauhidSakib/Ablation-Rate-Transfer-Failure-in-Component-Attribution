"""Compile PAPER.md to PDF via pandoc + XeLaTeX.

Three things need care and are handled here rather than left to chance:

1. The draft uses 42 distinct non-ASCII characters, most of them mathematical.
   Rather than hope a text font covers them, every one is mapped explicitly to a
   LaTeX construct via newunicodechar, so an unmapped glyph fails loudly at
   compile time instead of silently rendering as a blank box.
2. newunicodechar does not apply inside verbatim, so the two blocks that are
   actually mathematics are promoted to display math before pandoc sees them.
3. Typographic normalisation (R^2 -> a real superscript) is applied to prose
   only. Inside a code span the same string may be a literal identifier.
"""
from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

ROOT = Path(r"D:\naca")
OUT = ROOT / "paper"
SRC = ROOT / "PAPER.md"

CHI, PROD, ELEM, MINUS, MDOT, CAP = "\u03c7", "\u220f", "\u2208", "\u2212", "\u00b7", "\u2229"
SIGMA_U, HATG, COMB_HAT = "\u03a3", "\u011d", "\u0302"

WALSH_BLOCK = (
    "```\n"
    f"{CHI}_T(m) = {PROD}_{{i{ELEM}T}} ({MINUS}1)^{{m_i}} = {PROD}_g ({MINUS}1)^{{b_g{MDOT}|T {CAP} B_g|}} = {CHI}_{{S(T)}}(b),\n"
    f"        S(T) = {{ g : |T {CAP} B_g| is odd }}\n"
    "```"
)
WALSH_MATH = r"""$$\chi_T(m) \;=\; \prod_{i\in T}(-1)^{m_i}
\;=\; \prod_g (-1)^{\,b_g\cdot|T\cap B_g|} \;=\; \chi_{S(T)}(b),
\qquad S(T) = \bigl\{\, g : |T\cap B_g| \text{ is odd} \,\bigr\}$$"""

GHAT_LINE = f"> **{HATG}(S) = {SIGMA_U}_{{T : S(T)=S}} f{COMB_HAT}(T)**"
GHAT_MATH = r"$$\hat{g}(S) \;=\; \sum_{T \,:\, S(T)=S} \hat{f}(T)$$"


def _outside_code(text: str, fn):
    """Apply fn to prose only -- never inside fenced blocks or inline code spans."""
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        parts = re.split(r"(`[^`]*`)", line)
        out.append("".join(x if x.startswith("`") else fn(x) for x in parts))
    return "\n".join(out)


def preprocess(text: str) -> str:
    for name, block, repl in (("Walsh", WALSH_BLOCK, WALSH_MATH),
                              ("ghat", GHAT_LINE, GHAT_MATH)):
        if block in text:
            text = text.replace(block, repl, 1)
        else:
            print(f"  ! {name} block not matched; left as-is", file=sys.stderr)

    text = text.replace("f" + COMB_HAT, r"$\hat{f}$").replace(HATG, r"$\hat{g}$")
    # Any remaining letter followed by a combining circumflex becomes a proper
    # math accent; the bare combining mark has no glyph in either font.
    text = re.sub("([A-Za-z])" + COMB_HAT, r"$\\hat{\1}$", text)

    def norm(s: str) -> str:
        return s.replace("R^2", "R\u00b2").replace("2^9", "2\u2079")

    return _outside_code(text, norm)


UNI = {
    "\u03b1": r"\ensuremath{\alpha}",   "\u03c6": r"\ensuremath{\varphi}",
    "\u03c1": r"\ensuremath{\rho}",     "\u03c4": r"\ensuremath{\tau}",
    "\u03c0": r"\ensuremath{\pi}",      "\u03c3": r"\ensuremath{\sigma}",
    "\u03bb": r"\ensuremath{\lambda}",  "\u03c7": r"\ensuremath{\chi}",
    "\u03a3": r"\ensuremath{\Sigma}",
    "\u220f": r"\ensuremath{\prod}",    "\u2208": r"\ensuremath{\in}",
    "\u2229": r"\ensuremath{\cap}",     "\u222a": r"\ensuremath{\cup}",
    "\u2205": r"\ensuremath{\emptyset}", "\u2264": r"\ensuremath{\leq}",
    "\u2265": r"\ensuremath{\geq}",     "\u2260": r"\ensuremath{\neq}",
    "\u2248": r"\ensuremath{\approx}",  "\u221a": r"\ensuremath{\surd}",
    "\u211d": r"\ensuremath{\mathbb{R}}", "\u220e": r"\ensuremath{\blacksquare}",
    "\u2192": r"\ensuremath{\rightarrow}", "\u2016": r"\ensuremath{\|}",
    "\u00d7": r"\ensuremath{\times}",   "\u00b7": r"\ensuremath{\cdot}",
    "\u2212": r"\ensuremath{-}",        "\u00b1": r"\ensuremath{\pm}",
    "\u00b2": r"\ensuremath{^{2}}",     "\u00b9": r"\ensuremath{^{1}}",
    "\u2079": r"\ensuremath{^{9}}",     "\u207b": r"\ensuremath{^{-}}",
    "\u1d40": r"\ensuremath{^{\mathsf{T}}}", "\u1d62": r"\ensuremath{_{i}}",
    "\u2081": r"\ensuremath{_{1}}",     "\u2082": r"\ensuremath{_{2}}",
    "\u2714": r"\ensuremath{\checkmark}", "\u2020": r"\textdagger{}",
    # Added when 3.6 / 5.8 introduced new glyphs. An unmapped character is a
    # visible gap in the PDF, so map explicitly rather than trust the font --
    # note several of these appear inside code spans, where the MONO font has
    # even less coverage than the roman one.
    "\u03b2": r"\ensuremath{\beta}", "\u03b5": r"\ensuremath{\varepsilon}",
    "\u2074": r"\ensuremath{^{4}}", "\u2076": r"\ensuremath{^{6}}",
    "\u2026": r"\ldots{}", "\u0105": r"\k{a}",
}

PREAMBLE = r"""
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\setsansfont{Latin Modern Sans}
\setmonofont{Latin Modern Mono}[Scale=MatchLowercase]

\usepackage{amsmath,amssymb}
\usepackage{newunicodechar}
\usepackage{etoolbox}
\usepackage{booktabs,longtable,array}
\usepackage[table]{xcolor}
\usepackage{microtype}

% Tables here are numerous and some are wide. Shrink rather than overflow.
\AtBeginEnvironment{longtable}{\footnotesize}
\setlength{\LTpre}{0.8em}\setlength{\LTpost}{0.8em}

% Block quotes carry formal statements (propositions, rules); set them off.
\usepackage{framed}
\colorlet{shadecolor}{gray!12}
\renewenvironment{quote}{\begin{shaded}\vspace{-0.4em}}{\vspace{-0.4em}\end{shaded}}

\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue!60!black,citecolor=black]{hyperref}

\usepackage{graphicx}
\usepackage{float}
\usepackage[font=small,labelfont=bf]{caption}
% Figures are wide and information-dense; give them the full text width and let
% them float rather than splitting a table across the break.
\setkeys{Gin}{width=0.95\linewidth,keepaspectratio}

\usepackage{fvextra}
\fvset{breaklines=true,breakanywhere=true,fontsize=\small}

% Long \texttt paths (results/PREREG_wp7_seed2_full.md) have no natural break
% point and run into the margin. Permit a break after each underscore.
\let\NACAunderscore\_
\renewcommand{\_}{\NACAunderscore\allowbreak}
\sloppy
\emergencystretch=3em

__UNICHARS__
"""


def build() -> int:
    OUT.mkdir(exist_ok=True)
    text = preprocess(SRC.read_text(encoding="utf-8"))

    lines = text.split("\n")
    title = "Ablation-Rate Transfer Failure in Component Attribution"
    subtitle = "A Structurally Invisible Aggregate Term"
    if lines and lines[0].startswith("# "):
        full = lines[0][2:].strip()
        if ":" in full:
            title, subtitle = (s.strip() for s in full.split(":", 1))
        lines = lines[1:]

    md_path = OUT / "_paper_pp.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    unichars = "\n".join(r"\newunicodechar{%s}{%s}" % kv for kv in UNI.items())
    pre_path = OUT / "preamble.tex"
    pre_path.write_text(PREAMBLE.replace("__UNICHARS__", unichars), encoding="utf-8")

    import pypandoc

    args = [
        "--standalone", "--pdf-engine=xelatex",
        f"--include-in-header={pre_path}",
        "--toc", "--toc-depth=2",
        "-V", "documentclass=article", "-V", "papersize=a4",
        "-V", "fontsize=10pt", "-V", "geometry:margin=2.2cm",
        "-V", f"title={title}", "-V", f"subtitle={subtitle}",
        "-V", "date=" + datetime.date.today().strftime("%d %B %Y"),
        "-V", "linestretch=1.05", "-V", "colorlinks=true",
        "--syntax-highlighting=tango", "--wrap=preserve",
        # figure paths in PAPER.md are relative to the repo root
        f"--resource-path={ROOT}",
    ]
    pdf = OUT / "NACA_paper.pdf"
    pypandoc.convert_file(str(md_path), "pdf", format="markdown",
                          outputfile=str(pdf), extra_args=args)
    print(f"  OK  {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
