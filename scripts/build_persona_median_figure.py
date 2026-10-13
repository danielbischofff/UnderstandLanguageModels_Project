#!/usr/bin/env python3
"""Plot paired question medians and percentile bootstrap intervals per persona."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runtime' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import numpy as np

SEED = 20261007
SAMPLES = 10000
SETTINGS = dict(layer=62, scheme='layer-32', word_set_type='contextualized',
                policy='complete_only')
METHODS = [('band', 'mini-batch-cluster-readout', 'beta', .25),
           ('alpha', 'mini-alpha-analysis', 'alpha', 1.5)]
FIELDS = {'mrr': 'mean_reciprocal_rank_delta', 'cosine': 'activation_word_cosine_delta'}


def summaries():
    result = dict(configuration=SETTINGS, methods={}, sources={}, bootstrap={
        'estimator': 'Median of twelve answer-level paired question differences per persona.',
        'unit': 'Paired question; same sampled question IDs for both methods and all personas.',
        'samples': SAMPLES, 'seed': SEED,
        'interval': '2.5th and 97.5th percentiles of resampled medians; not intervals for means.',
        'limitations': 'Exploratory for twelve selected questions; fixed personas; no multiple-comparison correction.'})
    questions = None
    indices = None
    for name, folder, coefficient, strength in METHODS:
        source = ROOT / 'results' / folder / 'question-aggregation.json'
        data = json.loads(source.read_text())
        ids = sorted(q['id'] for q in data['questions'])
        if questions is None:
            questions = ids
            assert len(questions) == 12
            indices = np.random.default_rng(SEED).integers(0, len(ids), (SAMPLES, len(ids)))
        elif questions != ids:
            raise ValueError('Question sets must match.')
        rows = [r for r in data['per_question'] if all(r[k] == v for k, v in SETTINGS.items())
                and r[coefficient] == strength]
        lookup = {(r['character'], r['question']): r for r in rows}
        personas = sorted({r['character'] for r in rows})
        if len(personas) != 8 or len(rows) != 96 or len(lookup) != 96:
            raise ValueError('Require a complete, unique eight-persona/twelve-question grid.')
        own = dict(coefficient=coefficient, strength=strength, personas={})
        for persona in personas:
            metrics = {}
            for metric, field in FIELDS.items():
                paired = [lookup[persona, q] for q in questions]
                values = np.array([r[field] for r in paired], dtype=float)
                if not all(r['control_available'] for r in paired) or not np.isfinite(values).all():
                    raise ValueError('Missing or nonfinite paired score.')
                draws = np.median(values[indices], axis=1)
                metrics[metric] = dict(median=float(np.median(values)),
                                       ci95=np.quantile(draws, [.025, .975]).tolist(),
                                       n=12, question_deltas=dict(zip(questions, values.tolist())))
            own['personas'][persona] = metrics
        result['methods'][name] = own
        result['sources'][name] = dict(path=str(source.relative_to(ROOT)),
                                      sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    result['questions'] = questions
    return result


def plot(data, output):
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                         'axes.titlesize': 10, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    personas = list(data['methods']['band']['personas'])
    assert personas == list(data['methods']['alpha']['personas'])
    style = {'band': dict(color='#2563eb', marker='o', offset=-.16),
             'alpha': dict(color='#7c3aed', marker='D', offset=.16)}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.2), sharey=True)
    for ax, metric, label in zip(axes, ['mrr', 'cosine'], ['MRR', 'MCS']):
        all_stats = [data['methods'][m]['personas'][p][metric] for m in style for p in personas]
        bounds = [v for s in all_stats for v in [s['median'], *s['ci95']]]
        low, high = min(0, *bounds), max(0, *bounds)
        pad = (high-low or 1)*.09
        ax.set_xlim(low-pad, high+pad)
        for i in range(0, len(personas), 2):
            ax.axhspan(i-.5, i+.5, color='#64748b', alpha=.045, zorder=0)
        ax.axvline(0, color='#64748b', linewidth=.8, linestyle='--', zorder=1)
        for method, settings in style.items():
            for i, persona in enumerate(personas):
                stats = data['methods'][method]['personas'][persona][metric]
                y = i+settings['offset']
                # Draw intervals directly; percentile intervals need not enclose an estimator.
                ax.hlines(y, *stats['ci95'], color=settings['color'], linewidth=1.3, zorder=2)
                ax.vlines(stats['ci95'], y-.05, y+.05, color=settings['color'], linewidth=.9, zorder=2)
                ax.plot(stats['median'], y, marker=settings['marker'], color=settings['color'],
                        markersize=4, linestyle='none', zorder=3)
        ax.set_title(label, fontweight='bold', pad=6)
        ax.set_xlabel(f'Median paired $\\Delta${label}')
        ax.set_yticks(range(len(personas)), [p.capitalize() for p in personas])
        ax.set_ylim(len(personas)-.6, -.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, -3))
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(axis='x', color='#dce3eb', linewidth=.5, zorder=0)
        ax.tick_params(axis='y', length=0)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#b8c4d2')
    handles = [Line2D([0], [0], marker=style[m]['marker'], color=style[m]['color'],
                      linewidth=1.3, markersize=4, label=label)
               for m, label in [('band', r'Band-capping ($\beta=0.25$)'),
                                ('alpha', r'Alpha steering ($\alpha=1.5$)')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .99),
               ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.14, right=.98, bottom=.15, top=.85, wspace=.17)
    for suffix, options in [('pdf', {}), ('png', {'dpi': 250})]:
        fig.savefig(output.with_suffix('.'+suffix), bbox_inches='tight', **options)
    plt.close(fig)


def main():
    data = summaries()
    output = ROOT / 'writeup' / 'figures' / 'persona_median_comparison'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    plot(data, output)
    print(f'Created {output}.pdf, .png, and .json (8 personas, 2 methods, 2 metrics).')


if __name__ == '__main__':
    main()
