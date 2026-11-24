"""Render README charts from real results under results/. Never fabricates data:
charts whose inputs are missing are skipped with a message."""
import argparse
import json
import sys
from pathlib import Path

OFFICIAL = ['v1-batch', 'v2-streaming', 'v3-hosted', 'v4-hosted', 'v4-selfhosted']
SWEEP = ['eot-silence-300', 'eot-silence-500', 'eot-silence-700', 'eot-silence-1000',
         'eot-smart-0.3', 'eot-smart-0.5', 'eot-smart-0.7']


def find_latest(results_dir, prefix):
    """Newest results/<label>/results.json whose label == prefix or starts with prefix + '-'."""
    best = None
    for path in Path(results_dir).glob('*/results.json'):
        label = path.parent.name
        if label == prefix or label.startswith(prefix + '-'):
            if best is None or path.stat().st_mtime > best.stat().st_mtime:
                best = path
    return best


def load_latest(results_dir, prefix):
    path = find_latest(results_dir, prefix)
    if path is None:
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def metric(summary, name, stat='p50'):
    """Look up a summary value under the plausible key spellings; None when absent."""
    for key in (f'{name}_{stat}', f'{name}_ms_{stat}', f'{stat}_{name}', name):
        v = summary.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    # strip a trailing _ms from name and retry (endpoint_wait_ms -> endpoint_wait_p50)
    if name.endswith('_ms'):
        return metric(summary, name[:-3], stat)
    return None


def mean_endpoint_wait(result):
    """Mean endpoint_wait_ms over per-turn server metrics, if the results carry them."""
    vals = []
    for sc in result.get('scenarios', []):
        for key in ('turn_metrics', 'turns', 'server_turns'):
            for tm in sc.get(key, []) or []:
                d = tm.get('derived', tm) if isinstance(tm, dict) else {}
                v = d.get('endpoint_wait_ms')
                if isinstance(v, (int, float)):
                    vals.append(v)
    if vals:
        return sum(vals) / len(vals)
    return metric(result.get('summary', {}), 'endpoint_wait_ms', 'mean')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def latency_budget(results_dir, out_dir):
    rows = []
    for label in OFFICIAL:
        r = load_latest(results_dir, label)
        if r is None:
            print(f'latency_budget: no results for {label}', file=sys.stderr)
            continue
        s = r.get('summary', {})
        g = lambda n: metric(s, n) or 0.0
        parts = [
            ('endpoint wait', g('endpoint_wait_ms')),
            ('asr finalize', g('asr_finalize_ms')),
            ('llm ttft + tool', g('llm_ttft_ms') + g('tool_ms')),
            ('tts ttfa', g('tts_ttfa_ms')),
            ('send + network', g('send_ms') + g('network_ms')),
        ]
        rows.append((label, parts, metric(s, 'client_v2v', 'p50')))
    if not rows:
        print('latency_budget.png skipped: no official results found', file=sys.stderr)
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r[0] for r in rows]
    bottoms = [0.0] * len(rows)
    for i, stage in enumerate(p[0] for p in rows[0][1]):
        heights = [r[1][i][1] for r in rows]
        ax.bar(names, heights, bottom=bottoms, label=stage)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    cx = [(n, r[2]) for n, r in zip(names, rows) if r[2] is not None]
    if cx:
        ax.plot([c[0] for c in cx], [c[1] for c in cx], 'ko', label='client p50')
    ax.set_ylabel('p50 latency (ms)')
    ax.set_title('Voice-to-voice latency budget')
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / 'latency_budget.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def latency_vs_concurrency(results_dir, out_dir):
    series = {}
    for path in sorted(Path(results_dir).glob('loadtest-*.json')):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        levels = sorted((k for k in data if str(k).isdigit()), key=int)
        pts = [(int(k), data[k].get('client_v2v_p50'), data[k].get('client_v2v_p95')) for k in levels]
        if pts:
            series[path.stem.removeprefix('loadtest-')] = pts
    if not series:
        print('latency_vs_concurrency.png skipped: no results/loadtest-*.json', file=sys.stderr)
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, pts in series.items():
        for idx, label in ((1, 'p50'), (2, 'p95')):
            xy = [(p[0], p[idx]) for p in pts if p[idx] is not None]
            if xy:
                ax.plot(*zip(*xy), marker='o', linestyle='-' if idx == 1 else '--', label=f'{name} {label}')
    ax.set_xlabel('concurrent calls')
    ax.set_ylabel('client v2v (ms)')
    ax.set_title('Latency vs concurrency')
    ax.legend()
    fig.tight_layout()
    out = Path(out_dir) / 'latency_vs_concurrency.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def eot_tradeoff(results_dir, out_dir):
    pts = []
    for label in SWEEP:
        r = load_latest(results_dir, label)
        if r is None:
            continue
        x = mean_endpoint_wait(r)
        y = metric(r.get('summary', {}), 'false_cut_in_rate', 'p50')
        if x is not None and y is not None:
            pts.append((label, x, y * 100))
    if not pts:
        print('eot_tradeoff.png skipped: no sweep results found', file=sys.stderr)
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, x, y in pts:
        ax.scatter(x, y)
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('mean endpoint wait (ms)')
    ax.set_ylabel('false cut-in rate (%)')
    ax.set_title('End-of-turn tradeoff')
    fig.tight_layout()
    out = Path(out_dir) / 'eot_tradeoff.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='docs/img')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for fn in (latency_budget, latency_vs_concurrency, eot_tradeoff):
        out = fn(args.results, args.out)
        if out:
            print(f'wrote {out}')


if __name__ == '__main__':
    main()
