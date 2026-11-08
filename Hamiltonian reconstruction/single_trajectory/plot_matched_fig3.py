"""Compose Figure 3 from the coefficient reconstruction and matched fidelity data.

python3 plot_matched_fig3.py                   # render the stored figure data
python3 plot_matched_fig3.py --collect OUTPUT   # first check and use the fresh rerun
The included per-fit CSV supplies both protocols' fluctuating-interval results. For
each Hamiltonian the two replicates are evaluated separately, then their two
fidelities are averaged. Lines and bands summarize those 12 Hamiltonian means.
"""
import argparse,csv,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
STEM='trajectory_n10_matched'
DATA=HERE/'fig3_matched_fidelity.csv'
BUDGETS=[2000,5000,10000,20000,50000,100000,200000,500000]
def read(f):
    with f.open() as src:return list(csv.DictReader(src))
def collect(rerun):
    original={(int(r['hamiltonian_seed']),int(r['run']),int(r['rounds']),r['algorithm']):float(r['fidelity'])
              for r in read(HERE/'fig3_matched_runs.csv') if r['timing']=='fluctuating'}
    fresh={}
    for s in range(123,135):
        for r in read(rerun/f'seed_{s}.csv'):
            m=int(r['rounds']);rep=int(r['run']);key=(s,rep,m)
            assert key not in fresh and int(r['seed'])==s and int(r['preparations'])==m
            fresh[key]=float(r['fidelity'])
    assert len(fresh)==12*2*len(BUDGETS)
    diff=max(abs(v-original[k+('fresh',)]) for k,v in fresh.items())
    if diff>1e-10:raise ValueError(f'Fresh replay disagrees with archived scores: {diff}')
    rows=[]
    for m in BUDGETS:
        row=dict(rounds=m,thousands=m/1000)
        av={}
        for protocol in ['fresh','trajectory']:
            av[protocol]=np.array([np.mean([fresh[s,r,m] if protocol=='fresh' else original[s,r,m,protocol]
                                            for r in range(2)]) for s in range(123,135)])
            lo,med,hi=np.quantile(av[protocol],[.25,.5,.75])
            row.update({protocol+'_lo':lo,protocol+'_med':med,protocol+'_hi':hi})
        dif=np.sort(av['trajectory']-av['fresh'])
        row.update(paired_median_pp=100*np.median(dif),paired_ci_low_pp=100*dif[2],paired_ci_high_pp=100*dif[-3])
        rows.append(row)
    with DATA.open('w',newline='') as f:
        w=csv.DictWriter(f,rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Verified fresh estimates:',len(fresh),'maximum archived/replay difference:',diff)
    print(rows[-1])
def render():
    assert [int(r['rounds']) for r in read(DATA)]==BUDGETS
    with tempfile.TemporaryDirectory(prefix='matched_fig3_') as d:
        temp=Path(d);shutil.copyfile(DATA,temp/'data.csv')
        shutil.copyfile(HERE/'trajectory_n10_coefficients.pdf',temp/'coef.pdf')
        tex=r'''\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb,graphicx,pgfplots}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.18}
\definecolor{freshBlue}{HTML}{1f77b4}
\definecolor{chainOrange}{HTML}{ff7f0e}
\begin{document}\begin{tikzpicture}
\node[inner sep=0pt,anchor=north west] (left) at (0,0) {\includegraphics[height=8.05cm]{coef.pdf}};
\node[anchor=south,inner sep=1pt,font=\small] at ([yshift=0.10cm]left.north)
 {\textbf{(a)}\quad Coefficient reconstruction};
\begin{axis}[
 at={(8.0cm,-0.10cm)},anchor=north west,scale only axis,width=6.85cm,height=7.05cm,
 xmode=log,xmin=1.7,xmax=585,xtick={2,10,50,100,500},xticklabels={2,10,50,100,500},
 ymin=0,ymax=1,ytick={0,0.25,0.5,0.75,1},scaled ticks=false,
 xlabel={Measurement rounds $m$ (thousands)},ylabel={Coefficient fidelity $F$},
 tick align=inside,tick pos=both,major tick length=4pt,minor tick length=2pt,
 tick style={draw=black!45,line width=0.35pt},label style={font=\footnotesize},tick label style={font=\scriptsize},
 axis line style={draw=black!70,line width=0.45pt},legend pos=south east,
 legend style={font=\scriptsize,draw=black!55,line width=0.4pt,fill=white,cells={anchor=west}}]
'''
        styles=[('fresh','freshBlue','+','dashed','Fresh preparations'),('trajectory','chainOrange','x','solid','Single trajectory')]
        for alg,color,_,_,_ in styles:
            for k in ['lo','hi']:
                tex+=rf'\addplot[name path={alg}{k},draw=none,forget plot] table[x=thousands,y={alg}_{k},col sep=comma]{{data.csv}};'+'\n'
            tex+=rf'\addplot[{color},draw=none,fill opacity=.14,forget plot] fill between[of={alg}lo and {alg}hi];'+'\n'
        for alg,color,mark,ls,label in styles:
            tex+=rf'\addplot[{color},{ls},line width=.75pt,mark={mark},mark size=2.2pt,mark options={{solid,line width=.75pt}}] table[x=thousands,y={alg}_med,col sep=comma]{{data.csv}};'+'\n'
            tex+=rf'\addlegendentry{{{label}}}'+'\n'
        tex+=r'''\end{axis}
\node[anchor=south,inner sep=1pt,font=\small] at (11.425cm,0.10cm)
 {\textbf{(b)}\quad Fidelity versus measurements};
\end{tikzpicture}\end{document}
'''
        (temp/'figure.tex').write_text(tex)
        proc=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','figure.tex'],
                            cwd=temp,capture_output=True,text=True)
        if proc.returncode:raise RuntimeError(proc.stdout[-2500:])
        shutil.copyfile(temp/'figure.pdf',HERE/(STEM+'.pdf'))
    subprocess.run(['pdftoppm','-singlefile','-r','360','-png',str(HERE/(STEM+'.pdf')),str(HERE/STEM)],check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collect',type=Path);a=p.parse_args()
    if a.collect:collect(a.collect)
    render()
