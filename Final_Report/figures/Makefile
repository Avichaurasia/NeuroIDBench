SVG_FILES := $(wildcard *.svg)
INKSCAPE_FILES := $(SVG_FILES:%.svg=%.pdf_tex)

TEX_FILES := $(wildcard *.tex)
TIKZ_FILES := $(TEX_FILES:%.tex=%.pdf)

default: $(INKSCAPE_FILES) $(TIKZ_FILES)

%.pdf_tex: %.svg
	inkscape --export-area-drawing --export-latex -A $*.pdf $^

%.pdf: %.tex
	pdflatex $^
	pdfcrop --margins "5 5 5 5" $*.pdf $*.pdf
