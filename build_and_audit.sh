#!/bin/bash
set -u
cd /mnt/c/Users/soumi/tbnnFoam/docs/latex
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
bibtex main > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
echo "errors=$(grep -c '^!' main.log)"
echo "undefined=$(grep -c 'undefined' main.log)"
cp main.pdf ../tbnnFoam_preprint.pdf
rm -f tbnnFoam_arxiv_package.zip
python3 - <<'PYEOF'
import zipfile, os
os.chdir('/mnt/c/Users/soumi/tbnnFoam/docs/latex')
with zipfile.ZipFile('tbnnFoam_arxiv_package.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    z.write('main.tex'); z.write('references.bib'); z.write('main.bbl')
    for f in sorted(os.listdir('figures')):
        z.write(os.path.join('figures', f))
print('zip entries:', len(z.namelist()), sorted(os.listdir('figures')).__len__(), 'figures')
PYEOF
echo BUILD_DONE
