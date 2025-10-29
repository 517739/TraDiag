#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
usage: python count_trace_span.py  [可选结果csv路径，默认打印]
"""
import sys, pathlib, pandas as pd

def main():
    data_dir   = pathlib.Path('../Data')
    out_file   = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    summary    = []

    if not data_dir.is_dir():
        sys.exit(f'❌ 目录不存在: {data_dir.resolve()}')

    for csv_file in sorted(data_dir.rglob('*.csv')):
        try:
            df = pd.read_csv(csv_file)
            trace_cnt = df['TraceID'].nunique()
            span_cnt  = len(df)
            summary.append({'file': csv_file.as_posix(),
                            'trace_count': trace_cnt,
                            'span_count': span_cnt})
            print(f'{csv_file.name:<40}  {trace_cnt:>8} traces  {span_cnt:>10} spans')
        except Exception as e:
            print(f'⚠️  跳过 {csv_file}: {e}')

    if out_file:
        pd.DataFrame(summary).to_csv(out_file, index=False)
        print(f'\n→ 结果已写入 {out_file}')

if __name__ == '__main__':
    main()