"""Plot Supplementary Figure S5 from the accompanying coefficient CSV.

Run: python3 plot_coefficients.py
Requires pdflatex with PGFPlots. This renders only the single-trajectory PDF.
For the simulation of the selected trajectory, see simulate_coefficients.py.
"""
import csv
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
STEM = 'trajectory_n10_coefficients'

PREAMBLE = r'''\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb,pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{chainOrange}{HTML}{ff7f0e}
\pgfplotsset{paper/.style={
  width=7.6cm,height=6.5cm,
  tick align=inside,tick pos=both,major tick length=4pt,minor tick length=2pt,
  tick style={draw=black!45,line width=0.35pt},
  label style={font=\small},tick label style={font=\footnotesize},
  title style={font=\small,yshift=1pt},axis line style={draw=black!70,line width=0.45pt},
  legend style={font=\footnotesize,draw=black!55,line width=0.4pt,fill=white,cells={anchor=west}},
  scaled ticks=false}}
\begin{document}\begin{tikzpicture}
'''

END = r'\end{tikzpicture}\end{document}'

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def coefficients_figure(out,temp):
    path=out/'trajectory_n10_coefficients.csv'
    rows=read(path);assert len(rows)==81
    h=[float(r['h_true']) for r in rows];v=[float(r['h_reconstructed']) for r in rows]
    assert all(abs(sum(x*x for x in vector)-10)<1e-10 for vector in (h,v))
    lim=math.ceil(max(map(abs,h+v))*1.06*10)/10
    err=max(abs(float(r['residual'])) for r in rows)
    elim=math.ceil(1.12*err*100)/100
    shutil.copyfile(path,temp/'coeff.csv')
    return PREAMBLE+rf'''
\begin{{axis}}[paper,name=top,width=7.8cm,height=7.8cm,axis equal image,
  xmin=-{lim},xmax={lim},ymin=-{lim},ymax={lim},minor x tick num=1,minor y tick num=1,
  xlabel={{True coefficient $h_v$}},ylabel={{Reconstructed coefficient $\widehat h_v$}},legend pos=north west]
\addplot[black!45,dashed,line width=.65pt] coordinates {{(-{lim},-{lim}) ({lim},{lim})}};
\addlegendentry{{Identity}}
\addplot[only marks,chainOrange,mark=x,mark size=1.85pt,mark options={{line width=.7pt}}]
 table[x=h_true,y=h_reconstructed,col sep=comma]{{coeff.csv}};
\addlegendentry{{$F={float(rows[0]['fidelity']):.4f}$}}
\end{{axis}}
\begin{{axis}}[paper,at={{(top.south)}},anchor=north,yshift=-1.50cm,
 width=7.8cm,height=3.15cm,xmin=-1.5,xmax=81.5,ymin=-{elim},ymax={elim},
 xtick={{0,10,20,30,40,50,60,70,80}},xlabel={{Operator index $v$}},
 yticklabel style={{/pgf/number format/fixed,/pgf/number format/precision=2}},
 ylabel={{Error $\widehat h_v-h_v$}}]
\addplot[black!40,line width=.45pt] coordinates {{(-1.5,0) (81.5,0)}};
\addplot[only marks,chainOrange,mark=x,mark size=1.35pt,mark options={{line width=.60pt}}]
 table[x=index,y=residual,col sep=comma]{{coeff.csv}};
\end{{axis}}
'''+END

def main():
    with tempfile.TemporaryDirectory(prefix='single_probe_') as directory:
        temp = Path(directory)
        tex = temp / (STEM + '.tex')
        tex.write_text(coefficients_figure(HERE, temp))
        run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex.name],
                             cwd=temp, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stdout[-3000:] + run.stderr)
        shutil.copyfile(temp / (STEM + '.pdf'), HERE / (STEM + '.pdf'))

if __name__ == '__main__':
    main()
