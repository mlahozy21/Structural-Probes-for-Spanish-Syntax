"""Convert a CoNLL-U file into a whitespace-tokenized text file.

One line per sentence. Each token is the FORM column.

Three classes of CoNLL-U row need to be skipped to keep alignment with
mBERT inputs and with the surface-word view used elsewhere in the
pipeline:

  1. Comment lines (starting with '#').
  2. Contraction range rows (ID like '13-14', form like 'al'). The
     individual rows for the contracted pieces (13, 14) come right
     after and ARE kept, so the text gets 'a el' not 'al'.
  3. Enhanced empty nodes (ID like '8.1'). These have FORM='_' and
     represent syntactically projected but surface-absent tokens
     (e.g. dropped subjects in Spanish). They have no real text and
     mBERT cannot embed them; including them as '_' tokens corrupts
     subsequent alignment with the gold parse trees.

Both (2) and (3) are filtered with the same predicate: ID is not a pure
integer.

Usage
-----
  python -m scripts.conllu_to_text <input.conllu> <output.txt>
"""
from __future__ import annotations

import sys


def convert(conllu_path: str, output_path: str) -> None:
  print(f'Converting {conllu_path} -> {output_path}')
  written = 0
  skipped = 0
  with open(conllu_path, 'r', encoding='utf-8') as f_in, \
       open(output_path, 'w', encoding='utf-8') as f_out:
    tokens: list[str] = []
    for line in f_in:
      line = line.rstrip('\n')
      if not line.strip():
        if tokens:
          f_out.write(' '.join(tokens) + '\n')
          written += 1
          tokens = []
        continue
      if line.startswith('#'):
        continue
      parts = line.split('\t')
      if not parts[0].isdigit():
        skipped += 1
        continue
      if len(parts) >= 2:
        tokens.append(parts[1])
    if tokens:
      f_out.write(' '.join(tokens) + '\n')
      written += 1
  print(f'Wrote {written} sentences. Skipped {skipped} non-integer-ID '
        f'rows (range contractions + enhanced empty nodes).')


if __name__ == '__main__':
  if len(sys.argv) < 3:
    print('Usage: python -m scripts.conllu_to_text <input.conllu> <output.txt>')
    sys.exit(1)
  convert(sys.argv[1], sys.argv[2])
