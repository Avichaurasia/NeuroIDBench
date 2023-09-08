THESIS = thesis

short: 
	pdflatex $(THESIS).tex
	pdflatex $(THESIS).tex

complete: images lit short

lit: 
	pdflatex $(THESIS).tex
	bibtex $(THESIS)

images:
	make --directory=figures
