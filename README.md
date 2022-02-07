# text-tiling

This is an implementation of a simple TextTiling algorithm from Hearst (1997).
The current version only supports vocabulary introduction algorithm.

## Vocabulary Introduction

- `text_tile.py` will parse a raw text file, segment them into individual sentences if necessary, and segment them into subtopic segments.
- usage: `python text_tile.py [text-file-path] [--seq_size] [--num_bound] [--stem] [--segment] `
  - `[text-file-path]`: path to the raw textfile to be TextTiled
  - `[--seq_size] [-s]`: size of the token sequence. hyperparameter k in Hearst (1997).
  - `[--num_bound] [-n]`: number of subtopic segments to TextTile into. if left unspecified, it will be calculated from the distribution of scores.
  - `[--stem]`: stem each token if specified.
  - `[--segment]`: segment the text into real sentences. if left unspecified, it will be assumed that `\n` indicates sentence breaks. 

- example: `python text_tile.py witcher.txt  -s 20 -n 10 --stem`
  - This will run the program on a text file called `witcher.txt` with the token sequence size of 20, and tile them into 10 subtopic segments. Stemming will be performed before tiling.
