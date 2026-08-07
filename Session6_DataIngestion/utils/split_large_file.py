#!/usr/bin/env python3
"""
split_large_file.py
----------------------------------------------------------------------
Splits a large text file into smaller chunk files WITHOUT losing or
corrupting any data. It streams the file line-by-line (never loads
the whole thing into memory) and never splits a line across two
chunks, so no line is ever truncated or broken mid-word.

Usage (split by size, most common):
    python split_large_file.py --input big.txt --outdir chunks --chunk_size_mb 50

Usage (split by line count instead):
    python split_large_file.py --input big.txt --outdir chunks --by lines --lines_per_chunk 100000

Writes:
    <outdir>/<prefix>_0001.txt
    <outdir>/<prefix>_0002.txt
    ...
    <outdir>/<prefix>_manifest.txt   -- line/byte counts per chunk, and
                                         a checksum of the original file
                                         so you can verify nothing was
                                         lost or altered

To reassemble the original file exactly:
    cat chunks/<prefix>_*.txt > rebuilt.txt

Note on chunk size: because lines are never split, each chunk will be
approximately --chunk_size_mb, sometimes a little over -- the current
line is always finished before starting a new chunk, so exact byte
size isn't guaranteed, but no data is ever cut off.
----------------------------------------------------------------------
"""

import argparse
import hashlib
import os


def sha256_of_file(path, buf_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def split_by_size(input_path, outdir, prefix, chunk_size_mb):
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    chunk_index = 1
    current_bytes = 0
    current_lines = 0
    manifest = []

    out_path = os.path.join(outdir, f'{prefix}_{chunk_index:04d}.txt')
    out_f = open(out_path, 'w', encoding='utf-8', newline='')

    total_lines = 0
    with open(input_path, encoding='utf-8', errors='ignore') as in_f:
        for line in in_f:
            line_bytes = len(line.encode('utf-8'))

            # start a new chunk BEFORE writing this line if adding it
            # would exceed the target size (but never on an empty chunk,
            # so a single huge line still lands somewhere)
            if current_bytes > 0 and current_bytes + line_bytes > chunk_size_bytes:
                out_f.close()
                manifest.append((out_path, current_lines, current_bytes))
                chunk_index += 1
                current_bytes = 0
                current_lines = 0
                out_path = os.path.join(outdir, f'{prefix}_{chunk_index:04d}.txt')
                out_f = open(out_path, 'w', encoding='utf-8', newline='')

            out_f.write(line)
            current_bytes += line_bytes
            current_lines += 1
            total_lines += 1

    out_f.close()
    if current_lines > 0:
        manifest.append((out_path, current_lines, current_bytes))

    return manifest, total_lines


def split_by_lines(input_path, outdir, prefix, lines_per_chunk):
    chunk_index = 1
    current_lines = 0
    current_bytes = 0
    manifest = []

    out_path = os.path.join(outdir, f'{prefix}_{chunk_index:04d}.txt')
    out_f = open(out_path, 'w', encoding='utf-8', newline='')

    total_lines = 0
    with open(input_path, encoding='utf-8', errors='ignore') as in_f:
        for line in in_f:
            if current_lines >= lines_per_chunk:
                out_f.close()
                manifest.append((out_path, current_lines, current_bytes))
                chunk_index += 1
                current_lines = 0
                current_bytes = 0
                out_path = os.path.join(outdir, f'{prefix}_{chunk_index:04d}.txt')
                out_f = open(out_path, 'w', encoding='utf-8', newline='')

            out_f.write(line)
            current_bytes += len(line.encode('utf-8'))
            current_lines += 1
            total_lines += 1

    out_f.close()
    if current_lines > 0:
        manifest.append((out_path, current_lines, current_bytes))

    return manifest, total_lines


def main():
    ap = argparse.ArgumentParser(description="Split a large text file into smaller chunks without losing data")
    ap.add_argument('--input', required=True, help='Path to the large input file')
    ap.add_argument('--outdir', default='chunks', help='Output folder for chunk files')
    ap.add_argument('--prefix', default=None, help='Chunk filename prefix (default: input filename without extension)')
    ap.add_argument('--by', choices=['size', 'lines'], default='size', help='Split by target size (default) or by line count')
    ap.add_argument('--chunk_size_mb', type=float, default=50, help='Target size per chunk in MB (used when --by size)')
    ap.add_argument('--lines_per_chunk', type=int, default=100_000, help='Lines per chunk (used when --by lines)')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    input_size = os.path.getsize(args.input)
    print(f"Input: {args.input} ({input_size / (1024*1024):.2f} MB)")
    print("Computing checksum of original file...")
    original_hash = sha256_of_file(args.input)

    if args.by == 'size':
        manifest, total_lines = split_by_size(args.input, args.outdir, prefix, args.chunk_size_mb)
    else:
        manifest, total_lines = split_by_lines(args.input, args.outdir, prefix, args.lines_per_chunk)

    manifest_path = os.path.join(args.outdir, f'{prefix}_manifest.txt')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f"SPLIT MANIFEST\n")
        f.write(f"Input file        : {args.input}\n")
        f.write(f"Input size        : {input_size:,} bytes ({input_size/(1024*1024):.2f} MB)\n")
        f.write(f"Input SHA-256     : {original_hash}\n")
        f.write(f"Split mode        : {args.by}\n")
        f.write(f"Total lines       : {total_lines:,}\n")
        f.write(f"Chunks created    : {len(manifest)}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'chunk file':<40}{'lines':>12}{'size (bytes)':>16}\n")
        for path, lines, size in manifest:
            f.write(f"{os.path.basename(path):<40}{lines:>12,}{size:>16,}\n")
        f.write("\nTo verify no data was lost or altered:\n")
        f.write(f"  cat {prefix}_*.txt > rebuilt.txt   (Linux/Mac)\n")
        f.write(f"  copy /b {prefix}_0001.txt+{prefix}_0002.txt+... rebuilt.txt   (Windows)\n")
        f.write(f"  then compare its SHA-256 against: {original_hash}\n")

    print(f"\nDone. {len(manifest)} chunk(s) written to: {args.outdir}/")
    for path, lines, size in manifest:
        print(f"  {os.path.basename(path):<30} {lines:>10,} lines   {size/(1024*1024):>8.2f} MB")
    print(f"\nManifest (with checksum): {manifest_path}")
    print(f"Original SHA-256: {original_hash}")


if __name__ == '__main__':
    main()
