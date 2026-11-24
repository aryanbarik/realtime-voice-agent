"""Regenerate the results section of README.md from real results under results/.
Missing measurements are shown as 'not measured'; nothing is estimated."""
import argparse
import json
from pathlib import Path

from eval.charts import OFFICIAL, SWEEP, load_latest, mean_endpoint_wait, metric

START = '<!-- results:start -->'
END = '<!-- results:end -->'
NA = 'not measured'


def fmt(v, digits=0, pct=False, usd=False):
    if v is None:
        return NA
    if pct:
        return f'{v * 100:.1f}%'
    if usd:
        return f'${v:.4f}'
    return f'{v:.{digits}f}'


def _s(r):
    return r.get('summary', {}) if r else {}


def _g(r, name, stat='p50'):
    return metric(_s(r), name, stat) if r else None


def headline(res):
    best = res.get('v4-hosted')
    out = ['### Headline numbers', '']
    if not best:
        return out + [f'v4-hosted: {NA}. Run `uv run python -m eval.run_eval --profile v4-hosted`.', '']
    s = _s(best)
    out += [
        f"- v4-hosted client voice-to-voice p50: **{fmt(_g(best, 'client_v2v'))} ms**, "
        f"p95: {fmt(_g(best, 'client_v2v', 'p95'))} ms",
        f"- Task success: **{fmt(_g(best, 'task_success_rate', 'p50'), pct=True)}** "
        f"over {len(best.get('scenarios', [])) or NA} scenarios",
        f"- Cost per 3-minute call: {fmt(_g(best, 'cost_per_3min_call_usd', 'p50'), usd=True)}",
        '',
    ]
    return out


def profile_table(res):
    out = ['### Per-profile results', '',
           '| Profile | Success | Client v2v p50 | Client v2v p95 | Server v2v p50 | Endpoint wait | ASR finalize | '
           'LLM TTFT | Tool | TTS TTFA | Network | False cut-in | WER |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label in OFFICIAL:
        r = res.get(label)
        if not r:
            out.append(f'| {label} | ' + ' | '.join([NA] * 12) + ' |')
            continue
        out.append(
            f"| {label} | {fmt(_g(r, 'task_success_rate'), pct=True)} | {fmt(_g(r, 'client_v2v'))} | "
            f"{fmt(_g(r, 'client_v2v', 'p95'))} | {fmt(_g(r, 'server_v2v'))} | "
            f"{fmt(_g(r, 'endpoint_wait_ms'))} | {fmt(_g(r, 'asr_finalize_ms'))} | {fmt(_g(r, 'llm_ttft_ms'))} | "
            f"{fmt(_g(r, 'tool_ms'))} | {fmt(_g(r, 'tts_ttfa_ms'))} | {fmt(_g(r, 'network_ms'))} | "
            f"{fmt(_g(r, 'false_cut_in_rate'), pct=True)} | {fmt(_g(r, 'wer'), digits=3)} |")
    out += ['', 'Latencies in ms (p50 unless noted). v1-batch uses push-to-talk, so its endpoint wait is ~0, '
            'which flatters v1.', '']
    return out


def hosted_vs_self(res):
    out = ['### Hosted vs self-hosted (v4)', '',
           '| | v4-hosted | v4-selfhosted |', '|---|---:|---:|']
    h, s = res.get('v4-hosted'), res.get('v4-selfhosted')
    rows = [('Task success', 'task_success_rate', dict(pct=True), 'p50'),
            ('Client v2v p50 (ms)', 'client_v2v', {}, 'p50'),
            ('Client v2v p95 (ms)', 'client_v2v', {}, 'p95'),
            ('Barge-in client p50 (ms)', 'barge_in_client', {}, 'p50'),
            ('Cost per 3-min call', 'cost_per_3min_call_usd', dict(usd=True), 'p50')]
    for title, key, kw, stat in rows:
        out.append(f'| {title} | {fmt(_g(h, key, stat), **kw)} | {fmt(_g(s, key, stat), **kw)} |')
    out.append('')
    return out


def baseline_table(res):
    out = ['### Pipecat baseline', '', '| | Ours (v4-hosted) | Pipecat baseline |', '|---|---:|---:|']
    o, b = res.get('v4-hosted'), res.get('baseline-pipecat')
    for title, key, kw, stat in [('Task success', 'task_success_rate', dict(pct=True), 'p50'),
                                 ('Client v2v p50 (ms)', 'client_v2v', {}, 'p50'),
                                 ('Client v2v p95 (ms)', 'client_v2v', {}, 'p95'),
                                 ('False cut-in rate', 'false_cut_in_rate', dict(pct=True), 'p50')]:
        out.append(f'| {title} | {fmt(_g(o, key, stat), **kw)} | {fmt(_g(b, key, stat), **kw)} |')
    out += ['', 'Server-stage metrics are n/a for the baseline (only client-side metrics are comparable).', '']
    return out


def loadtest_table(results_dir):
    out = ['### Load test', '', '| Profile | Level | Client p50 (ms) | Client p95 (ms) | Success | Errors |',
           '|---|---:|---:|---:|---:|---:|']
    n = 0
    for path in sorted(Path(results_dir).glob('loadtest-*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        name = path.stem.removeprefix('loadtest-')
        for level in sorted((k for k in data if str(k).isdigit()), key=int):
            d = data[level]
            err = d.get('errors')
            out.append(f"| {name} | {level} | {fmt(d.get('client_v2v_p50'))} | {fmt(d.get('client_v2v_p95'))} | "
                       f"{fmt(d.get('success_rate'), pct=True)} | {err if err is not None else NA} |")
            n += 1
    if n == 0:
        return ['### Load test', '', f'{NA}. Run `uv run python -m eval.loadtest`.', '']
    return out + ['']


def sweep_table(res):
    out = ['### End-of-turn sweep', '',
           '| Setting | Mean endpoint wait (ms) | False cut-in rate | Task success |', '|---|---:|---:|---:|']
    n = 0
    for label in SWEEP:
        r = res.get(label)
        if r:
            n += 1
        out.append(f"| {label} | {fmt(mean_endpoint_wait(r) if r else None)} | "
                   f"{fmt(_g(r, 'false_cut_in_rate'), pct=True)} | {fmt(_g(r, 'task_success_rate'), pct=True)} |")
    if n == 0:
        return ['### End-of-turn sweep', '', f'{NA}. Run `uv run python -m eval.sweep_eot`.', '']
    return out + ['']


def render(results_dir):
    res = {label: load_latest(results_dir, label)
           for label in OFFICIAL + ['baseline-pipecat'] + SWEEP}
    parts = headline(res) + profile_table(res) + hosted_vs_self(res) + baseline_table(res) \
        + loadtest_table(results_dir) + sweep_table(res)
    return '\n'.join(parts).rstrip() + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--readme', default='README.md')
    args = ap.parse_args()
    path = Path(args.readme)
    text = path.read_text(encoding='utf-8')
    if START not in text or END not in text:
        raise SystemExit(f'{path} is missing {START} / {END} markers')
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    path.write_text(f'{head}{START}\n{render(args.results)}{END}{tail}', encoding='utf-8')
    print(f'updated {path}')


if __name__ == '__main__':
    main()
