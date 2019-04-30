.PHONY=zip

report.pdf: README.md
	pandoc README.md -o report.pdf 

zip: report.pdf
	cd ../ && zip 20161105-pegasos.zip -r pegasos-svm/*.py pegasos-svm/*.md  pegasos-svm/*.pdf pegasos-svm/makefile pegasos-svm/.gitignore pegasos-svm/data/README pegasos-svm/.gitignore  pegasos-svm/report.pdf
