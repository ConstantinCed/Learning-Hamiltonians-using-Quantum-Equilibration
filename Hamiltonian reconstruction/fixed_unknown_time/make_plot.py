"""Create the fixed-time recovery figure from completed simulation records.

Run after run_sweep.py, or use --data ../../output/fixed_time_recovery_1m
after run_million.py. PDF/PNG, per-instance metrics, and plot CSVs are saved
beside this script. Requires pdflatex with PGFPlots and pdftoppm (Poppler).
Use --view early for a separate linear-axis view of the measured times t<=10.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent
STEM='fixed_unknown_time'


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def main(data,view='full'):
    stem=STEM if view=='full' else STEM+'_early_time'
    if (data/'metadata.json').exists():
        config=json.loads((data/'metadata.json').read_text())['config']
        source_metrics=data/'metrics.csv'
    else:
        source_metrics=HERE/'metrics.csv'
        with source_metrics.open() as f:available=list(csv.DictReader(f))
        sizes=sorted({int(r['n']) for r in available})
        shots={int(r['total_shots']) for r in available}
        assert len(shots)==1
        config=dict(sizes=sizes,shots=shots.pop(),
            instances=len({int(r['seed']) for r in available if int(r['n'])==sizes[0]}),
            first_seed=min(int(r['seed']) for r in available),
            times=sorted({float(r['time']) for r in available}))
    with source_metrics.open() as f:metrics=list(csv.DictReader(f))
    assert len(metrics)==len(config['sizes'])*config['instances']*len(config['times'])
    write_csv(HERE/'metrics.csv',metrics)
    summaries=[]
    for n in config['sizes']:
        for t in config['times']:
            rows=[r for r in metrics if int(r['n'])==n and float(r['time'])==t]
            assert len(rows)==config['instances']
            item=dict(n=n,time=t,total_shots=config['shots'],instances=config['instances'])
            for name in ['direction_error','fidelity','pi_U','max_non_h_autocorrelation']:
                values=[float(r[name]) for r in rows]
                lo,med,hi=np.percentile(values,[25,50,75])
                item.update({name:med,name+'_q25':lo,name+'_q75':hi})
            summaries.append(item)
        write_csv(HERE/f'plot_n{n}.csv',[r for r in summaries if r['n']==n])
        if view=='early':
            write_csv(HERE/f'plot_early_n{n}.csv',
                [r for r in summaries if r['n']==n and r['time']<=10])
    write_csv(HERE/'summary.csv',summaries)
    previous=data.parent/'fixed_time_recovery/metrics.csv'
    if config['shots']==1000000 and previous.exists():
        with previous.open() as f:old=list(csv.DictReader(f))
        comparison=[]
        for row in summaries:
            selected=[r for r in old if int(r['n'])==row['n'] and float(r['time'])==row['time']]
            assert len(selected)==config['instances']
            assert all(int(r['total_shots'])==100000 for r in selected)
            old_error=float(np.median([float(r['direction_error']) for r in selected]))
            comparison.append(dict(n=row['n'],time=row['time'],
                error_100k=old_error,error_1m=row['direction_error'],
                error_reduction_factor=old_error/row['direction_error'],
                fidelity_100k=float(np.median([float(r['fidelity']) for r in selected])),
                fidelity_1m=row['fidelity']))
        write_csv(HERE/'budget_comparison.csv',comparison)
    source=r'''\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots,fillbetween}
\pgfplotsset{compat=1.18}
\definecolor{cSix}{HTML}{1f77b4}
\definecolor{cEight}{HTML}{2ca02c}
\definecolor{cTen}{HTML}{d62728}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[
group style={group size=2 by 1,horizontal sep=1.7cm},
width=7.6cm,height=6.5cm,
xmode=log,ymode=log,xmin=0.008,xmax=12500,
xtick={0.01,1,100,10000},
tick align=inside,tick pos=both,major tick length=4pt,minor tick length=2pt,
tick style={draw=black!45,line width=0.35pt},
label style={font=\small},tick label style={font=\footnotesize},
title style={font=\small,yshift=1pt},axis line style={draw=black!70,line width=0.45pt},
xlabel={Evolution time $t$},
legend style={font=\footnotesize,draw=black!50,line width=0.4pt,fill=white,cells={anchor=west}},
]
\nextgroupplot[title={\textbf{a}\quad Reconstruction across time},
ylabel={Reconstruction error},ymin=0.075,ymax=1.7,
ytick={0.1,0.2,0.5,1},yticklabels={0.1,0.2,0.5,1},legend pos=north east]
ERROR_PLOTS
\nextgroupplot[title={\textbf{b}\quad Isolation of the Hamiltonian},
ylabel={Dynamical gap $\pi_U(t)$},ymin=0.000001,ymax=2.4]
GAP_PLOTS
\end{groupplot}
\end{tikzpicture}
\end{document}
'''
    def panels(quantity,legend):
        commands=[]
        for n,color in zip(config['sizes'],['cSix','cEight','cTen']):
            tag=f'{quantity}{n}'
            table=f'plot_n{n}.csv' if view=='full' else f'plot_early_n{n}.csv'
            commands.extend([
                rf'\addplot[name path={tag}lo,draw=none,forget plot] table[x=time,y={quantity}_q25,col sep=comma]{{{table}}};',
                rf'\addplot[name path={tag}hi,draw=none,forget plot] table[x=time,y={quantity}_q75,col sep=comma]{{{table}}};',
                rf'\addplot[{color},fill opacity=0.12,draw=none,forget plot] fill between[of={tag}lo and {tag}hi];',
                rf'\addplot[{color},line width=0.7pt,mark=x,mark size=2.0pt,mark options={{line width=0.7pt}}] table[x=time,y={quantity},col sep=comma]{{{table}}};'])
            if legend:commands.append(rf'\addlegendentry{{$n={n}$}}')
        return '\n'.join(commands)
    # Bounds follow all IQR bands, without truncating displayed data.
    error_min=min(r['direction_error_q25'] for r in summaries)
    error_floor=0.02 if error_min>=0.025 else 10**np.floor(np.log10(error_min/1.2))
    source=source.replace('ymin=0.075,ymax=1.7',f'ymin={error_floor:.9g},ymax=1.7')
    source=source.replace('ytick={0.1,0.2,0.5,1},yticklabels={0.1,0.2,0.5,1}',
        'ytick={0.02,0.05,0.1,0.2,0.5,1},yticklabels={0.02,0.05,0.1,0.2,0.5,1}')
    minimum=min(r['pi_U_q25'] for r in summaries)
    gap_min=10**np.floor(np.log10(minimum))
    source=source.replace('ymin=0.000001,ymax=2.4',f'ymin={gap_min:.9g},ymax=2.4')
    if view=='early':
        source=source.replace('xmode=log,ymode=log,xmin=0.008,xmax=12500,',
            'xmode=normal,ymode=normal,xmin=0,xmax=10.15,')
        source=source.replace('xtick={0.01,1,100,10000},','xtick={0,2,4,6,8,10},')
        source=source.replace(f'ymin={error_floor:.9g},ymax=1.7',
            'ymin=0,ymax=1.5')
        source=source.replace('ytick={0.02,0.05,0.1,0.2,0.5,1},yticklabels={0.02,0.05,0.1,0.2,0.5,1}',
            'ytick={0,0.5,1,1.5},yticklabels={0,0.5,1,1.5}')
        source=source.replace(f'ymin={gap_min:.9g},ymax=2.4',
            'ymin=0,ymax=2,ytick={0,0.5,1,1.5,2}')
        source=source.replace('Reconstruction across time','Onset of reconstruction')
        source=source.replace('Isolation of the Hamiltonian','Opening of the dynamical gap')
    source=source.replace('ERROR_PLOTS',panels('direction_error',True)).replace('GAP_PLOTS',panels('pi_U',False))
    with tempfile.TemporaryDirectory(prefix='fixed_time_figure_') as directory:
        temp=Path(directory);tex=temp/f'{stem}.tex';tex.write_text(source)
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',f'-output-directory={temp}',str(tex)],cwd=HERE,check=True,capture_output=True)
        shutil.copy2(temp/f'{stem}.pdf',HERE/f'{stem}.pdf')
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(HERE/f'{stem}.pdf'),str(HERE/stem)],check=True)
    late=[]
    for n in config['sizes']:
        rows=[r for r in metrics if int(r['n'])==n and float(r['time'])>=10]
        late.append(dict(n=n,median_fidelity_across_instances_and_late_times=float(np.median([float(r['fidelity']) for r in rows])),
            minimum_late_time_median_fidelity=min(r['fidelity'] for r in summaries if r['n']==n and r['time']>=10),
            maximum_late_time_median_fidelity=max(r['fidelity'] for r in summaries if r['n']==n and r['time']>=10)))
    report=dict(config=config,late_time_summary=late,
        max_conservation_residual=max(float(r['conservation_residual']) for r in metrics),
        max_probability_normalization_error=max(float(r['probability_norm_error']) for r in metrics))
    data.mkdir(parents=True,exist_ok=True)
    report.update(view=view,displayed_time_range=[0,10] if view=='early' else [0.01,10000],
        displayed_axes='linear' if view=='early' else 'logarithmic')
    (data/('figure_report.json' if view=='full' else 'figure_report_early_time.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=HERE.parents[1]/'output/fixed_time_recovery')
    parser.add_argument('--view',choices=['full','early'],default='full')
    args=parser.parse_args();main(args.data,args.view)
