.PHONY=zip

report.pdf: README.md
	pandoc README.md -o report.pdf

zip:
	cd ../ && zip 20161105-pegasos.zip -R pegasos/
