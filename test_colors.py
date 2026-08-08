import re

with open('assets/portrait_svg.svg', 'r') as f:
    content = f.read()
fills = re.findall(r'fill=\"([^\"]+)\"', content)
print('Number of unique colors:', len(set(fills)))
print('Sample colors:', list(set(fills))[:10])
