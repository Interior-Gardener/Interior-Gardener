import xml.etree.ElementTree as ET

tree = ET.parse('dark.svg')
root = tree.getroot()
main_g = root.find('{http://www.w3.org/2000/svg}g')
for i, child in enumerate(main_g):
    tag = child.tag.split('}')[-1]
    if tag == 'g' and 'id' in child.attrib:
        print(f'{i}: {tag} id={child.attrib["id"]} class={child.attrib.get("class", "")}')
    else:
        print(f'{i}: {tag} attr: {child.attrib}')
