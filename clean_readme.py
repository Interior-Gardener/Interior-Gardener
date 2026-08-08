import re

mojibake_map = {
    "≡ƒæ¿ΓÇì≡ƒÆ╗": "👨‍💻",
    "≡ƒÜÇ": "🚀",
    "≡ƒÄ»": "🎯",
    "≡ƒÅå": "🏆",
    "≡ƒÆ¼": "💬",
    "Γ¡É": "⭐",
    "≡ƒîƒ": "🌟",
    "≡ƒîì": "🌍",
    "Γ£¿": "✨",
    "≡ƒ¢á": "🛠️",
    "≡ƒÅà": "🏅",
    "≡ƒº¬": "🧪",
    "ΓÜ¢": "⚛️",
    "≡ƒôÜ": "📚",
    "≡ƒºá": "🧠",
    "≡ƒÄ«": "🎓",
    "≡ƒÑç": "🥇",
    "≡ƒÑê": "🥈", # AI-Robo Festival Runner-up
    "≡ƒÅ¡": "🏭",
    "Γ£ö": "✔️",
    "≡ƒñ¥": "🤝",
    "≡ƒôê": "📈",
    "≡ƒÆ│": "💳",
    "≡ƒöì": "🔍",
    "≡ƒôü": "📁",
    "≡ƒÆí": "💡",
    "≡ƒæÑ": "👥",
    "≡ƒÆ╝": "💼",
    "≡ƒÄô": "🏫",
    "≡ƒôì": "📍",
    "≡ƒôà": "📅",
    "≡ƒÜå": "🚆",
    "≡ƒñû": "🤖",
    "≡ƒÅ╕": "🏸",
    "Γö£ΓöÇΓöÇ": "├──",
    "Γöé": "│",
    "ΓööΓöÇΓöÇ": "└──",
    "ΓÇö": "—",
    "ΓÇó": "•",
    "┬╖": "·",
    "≡ƒöù": "🔗",
    "≡ƒç«≡ƒç│": "🇮🇳",
    "≡ƒÉì": "🐍",
    "≡ƒùä": "🗄️",
    "ΓÜÖ": "⚙️",
    "Γÿü": "☁️",
    "Γÿò": "☕",
    "≡ƒîÖ": "🌙",
    "≡ƒî▒": "🌱",
    "≡ƒôè": "📊",
    "≡ƒôî": "📌",
    "≡ƒôû": "📖",
    "≡ƒô£": "📜",
    "∩╕Å": "", # Just an invisible modifier sometimes appended to gear/cloud
    "├ù": "×" # YuWaah ├ù UNICEF
}

with open("README.backup.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

authentic_content = "".join(lines[996:]) # Lines 997 to end

# Replace all mojibake
for moji, emoji in mojibake_map.items():
    authentic_content = authentic_content.replace(moji, emoji)

with open("clean_content.md", "w", encoding="utf-8") as f:
    f.write(authentic_content)
