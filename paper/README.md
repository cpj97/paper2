# Paper Scaffold

This folder is the writing workspace for the manuscript.

## Suggested workflow

1. Write the paper in `main.tex` and the files in `sections/`.
2. Keep Zotero as the source of truth for references.
3. Auto-export the Zotero library or a Zotero collection to `paper/references/zotero.bib`.
4. Move manuscript-ready figures into `paper/figures/` only when you need hand-edited versions.
5. Otherwise, include figures directly from `../outputs/figures/` or `../outputs/maps/`.
6. Store manuscript-ready regression tables in `paper/tables/`.

## Zotero setup

Recommended setup:

- Install Better BibTeX in Zotero.
- Create a dedicated collection for this paper.
- Auto-export that collection to `paper/references/zotero.bib`.
- Use stable citation keys in Zotero so your LaTeX citations do not drift.

The LaTeX scaffold uses `biblatex` with `biber`, which is the cleanest option for Zotero and Better BibTeX.

## Figures and tables

The scaffold already points `\graphicspath` to:

- `paper/figures/`
- `../outputs/figures/`
- `../outputs/maps/`

That lets you keep analysis outputs in the project pipeline while reserving `paper/figures/` for final edited versions.

## Build

If a TeX toolchain is installed, a standard build command is:

```powershell
latexmk -pdf -outdir=build main.tex
```

This environment does not currently have a working TeX compiler available, so I could not verify a full compile here.
