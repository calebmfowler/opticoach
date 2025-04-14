from collections import defaultdict

# Input data (replace this with reading from the file if needed)
lines = [
    "East Texas Baptist",
    "Texas College",
    "North Texas State",
    "West Texas State",
    "Texas Christian",
    "Texas HS (TX)",
    "Texas Western / UTEP",
    "Texas Military College",
    "Athens High School (Texas)",
    "Texas A&I",
    "Texas A&I / Texas A&M–Kingsville",
    "North Texas State / North Texas",
    "North Texas State Teachers",
    "Texas A&M (GA)",
    "Texas Lutheran",
    "Texas Mines",
    "Texas Western",
    "West Texas State Teachers",
    "Texas Wesleyan",
    "Texas Terror",
    "East Texas State",
    "Texas A&M Aggies",
    "Texas Tech Red Raiders",
    "Southwest Texas St.",
    "West Texas Rufneks",
    "Texas City HS",
    "Texas–Arlington (head coach)",
    "Texas Southern Tigers",
    "Texas–El Paso",
    "North Texas State Normal/Teachers",
    "North Texas State Normal",
    "Jacksonville High (Texas)",
    "Texas State / Texas Southern",
    "Texas Mines / Texas Western",
    "Texas A&M-Kingsville",
    "Texas A&M-Commerce",
    "North Texas Aggies / Arlington State"
]

# Grouping logic
grouped_teams = defaultdict(list)

for line in lines:
    # Split by common delimiters like "/", "(", etc.
    parts = [part.strip() for part in line.replace("/", ",").replace("(", ",").replace(")", "").split(",")]
    for part in parts:
        grouped_teams[part].append(line)

# Output grouped results
for team, references in grouped_teams.items():
    print(f"{team}: {references}")