COMPILER = pdflatex -shell-escape

pdf:
	$(COMPILER) main.tex
	biber main
	$(COMPILER) main.tex
	$(COMPILER) main.tex

copy:
	rm -rf out
	cp -r ../code/out .

clean:
	rm -rf *.bbl *.aux *.bcf *.blg *.log *.run.xml *.out *.tox

