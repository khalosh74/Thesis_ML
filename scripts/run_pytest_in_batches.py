#!/usr/bin/env python3
import glob
import subprocess
import sys
import math
import argparse


def main(batch_size: int = 12) -> int:
    files = sorted(glob.glob('tests/**/test_*.py', recursive=True))
    if not files:
        print('No test files found under tests/')
        return 0
    total = len(files)
    print(f'Found {total} test files. Running in batches of {batch_size}.')
    failed_batches = []
    batch_count = math.ceil(total / batch_size)
    for i in range(0, total, batch_size):
        chunk = files[i : i + batch_size]
        batch_idx = (i // batch_size) + 1
        print('\n' + '=' * 80)
        print(f'Batch {batch_idx}/{batch_count}: running {len(chunk)} files')
        print(' '.join(chunk))
        cmd = [sys.executable, '-m', 'pytest', '-q'] + chunk
        rc = subprocess.run(cmd)
        if rc.returncode != 0:
            print(f'Batch {batch_idx} returned exit code {rc.returncode}')
            failed_batches.append((batch_idx, rc.returncode))
    print('\n' + '=' * 80)
    if failed_batches:
        print(f'{len(failed_batches)} batch(es) failed: {failed_batches}')
        return 1
    print('All batches completed with exit code 0')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', type=int, default=12)
    args = parser.parse_args()
    sys.exit(main(args.batch_size))
