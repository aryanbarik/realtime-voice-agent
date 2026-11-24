import argparse
import glob
import json
from pathlib import Path

import numpy as np

METRICS = ('v2v_ms', 'endpoint_wait_ms', 'asr_finalize_ms', 'llm_ttft_ms', 'tool_ms',
           'llm_first_clause_ms', 'tts_ttfa_ms', 'send_ms', 'barge_in_ms')


def report(paths):
    values = {name: [] for name in METRICS}
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                for name, value in item.get('derived', {}).items():
                    if name in values and value is not None:
                        values[name].append(value)
    rows = ['| Metric | Count | p50 (ms) | p95 (ms) |', '|---|---:|---:|---:|']
    for name, vals in values.items():
        if vals:
            p50, p95 = np.percentile(vals, [50, 95], method='linear')
            rows.append(f'| {name} | {len(vals)} | {p50:.1f} | {p95:.1f} |')
        else:
            rows.append(f'| {name} | 0 | n/a | n/a |')
    return '\n'.join(rows) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('traces', nargs='+')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    paths = [p for pattern in args.traces for p in (glob.glob(pattern) or [pattern])]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report(paths), encoding='utf-8')


if __name__ == '__main__':
    main()
