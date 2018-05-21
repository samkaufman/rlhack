#!/usr/bin/env python3

import os
import re

def main():
    all_scores = {}
    for dirpath, _, filenames in os.walk("./result"):
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            fullpath = os.path.join(dirpath, filename)
            with open(fullpath, 'r') as fo:
                lines = fo.readlines()
            scores = []
            for line in lines:
                m = re.match(r"^Correct: (\d+)\/(\d+).*$", line)
                if m:
                    scores.append(int(m.group(1))/int(m.group(2)))
            all_scores[filename] = max(scores)

    best_score = max(all_scores.values())
    for k, v in all_scores.items():
        if v > best_score * 0.95:
            print(f"{k}: {v}")

if __name__ == '__main__':
    main()