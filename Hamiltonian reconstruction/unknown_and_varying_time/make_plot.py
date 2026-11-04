"""Rebuild the figure from coefficients.csv and fidelity.csv.

Usage: python3 make_plot.py
Requires pdflatex with PGFPlots and pdftoppm (Poppler). Intermediate LaTeX files
are created in a temporary directory; only the PDF and PNG are saved here.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
STEM = 'unknown_and_varying_time'
TEX = '\\documentclass[border=3pt]{standalone}\n\\usepackage{amsmath,amssymb}\n\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots}\n\\pgfplotsset{compat=1.18}\n\\definecolor{cX}{HTML}{1f77b4}\n\\begin{document}\n\\begin{tikzpicture}\n\\begin{groupplot}[\ngroup style={group size=2 by 1,horizontal sep=1.6cm},\nwidth=7.6cm,height=6.5cm,\ntick align=inside,tick pos=both,major tick length=4pt,minor tick length=2pt,\ntick style={draw=black!45,line width=0.35pt},\nlabel style={font=\\small},tick label style={font=\\footnotesize},\ntitle style={font=\\small,yshift=1pt},axis line style={draw=black!70,line width=0.45pt},\n]\n\\nextgroupplot[\ntitle={\\textbf{a}\\quad Recovery under unknown timing},\naxis equal image,xmin=-0.320,xmax=0.320,ymin=-0.320,ymax=0.320,\nminor x tick num=1,minor y tick num=1,\nxlabel={True coefficient $h_\\alpha$},ylabel={Reconstructed coefficient $\\widehat h_\\alpha$},\n]\n\\addplot[black!45,dashed,line width=0.65pt,domain=-0.320:0.320,samples=2]{x};\n\\addplot[only marks,mark=x,mark size=1.9pt,mark options={line width=0.65pt},cX] table[x=true_h,y=spectral,col sep=comma]{coefficients.csv};\n\n\\nextgroupplot[\ntitle={\\textbf{b}\\quad Finite-shot recovery},\nxmode=log,xmin=0.85,xmax=114.99999999999999,xtick={1,5,20,50,100},xticklabels={1,5,20,50,100},\nymin=0,ymax=100,ytick={0,25,50,75,100},\nxlabel={Total experimental shots (thousands)},ylabel={Recovery fidelity (\\%)},\n]\n\\addplot[cX,line width=0.65pt,mark=x,mark size=2.4pt,mark options={line width=0.8pt},\nerror bars/.cd,y dir=both,y explicit,error bar style={line width=0.55pt},error mark options={rotate=90,mark size=2.5pt}]\ntable[x=thousands,y=fidelity_percent,y error minus=lower,y error plus=upper,col sep=comma]{fidelity.csv};\n\n\\end{groupplot}\n\\end{tikzpicture}\n\\end{document}\n'


def main():
    with tempfile.TemporaryDirectory(prefix='varying_time_plot_') as directory:
        temp = Path(directory)
        source = temp / f'{STEM}.tex'
        source.write_text(TEX)
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                        f'-output-directory={temp}', str(source)],
                       cwd=HERE, check=True, capture_output=True)
        shutil.copy2(temp / f'{STEM}.pdf', HERE / f'{STEM}.pdf')
    subprocess.run(['pdftoppm', '-singlefile', '-r', '200', '-png',
                    str(HERE / f'{STEM}.pdf'), str(HERE / STEM)], check=True)


if __name__ == '__main__':
    main()
