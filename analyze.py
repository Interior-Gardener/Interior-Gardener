import collections
import re

with open('README.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

headers = []
for i, line in enumerate(lines):
    if line.startswith('#'):
        headers.append(f"{i+1}: {line.strip()}")

with open('analysis_output.txt', 'w', encoding='utf-8') as out:
    out.write("--- HEADERS ---\n")
    for h in headers:
        out.write(h + "\n")

    # Find duplicates
    header_counts = collections.Counter([h.split(': ', 1)[1] for h in headers])
    out.write("\n--- DUPLICATED HEADERS ---\n")
    for k, v in header_counts.items():
        if v > 1:
            out.write(f"{k}: {v} times\n")

    out.write("\n--- MOJIBAKE MATCHES ---\n")
    for i, line in enumerate(lines):
        if re.search(r'[≡ΓƒÃÂ]', line):
            out.write(f"Line {i+1}: {line.strip()}\n")
