import re

with open('assets/portrait_svg.svg', 'r') as f:
    content = f.read()

paths = re.findall(r'<path fill=\"([^\"]+)\" d=\"([^\"]+)\"', content)
print(len(paths), 'paths found.')

parsed_pts = []
for fill, d in paths:
    # d looks like M0 0h1v1H0zM5 0h1v1H5z
    commands = d.split('z')
    for cmd in commands:
        if not cmd: continue
        # Find M x y
        m = re.search(r'M(\d+)\s*(\d+)', cmd)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            parsed_pts.append((x, y, fill))

print('Total dots parsed:', len(parsed_pts))
