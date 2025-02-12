import pandas as pd

def encode_yardsToGo(yards_to_go):
    if pd.isna(yards_to_go):
        return -1

    if yards_to_go == 1: # Inches (1 yard to go)
        return 0
    elif yards_to_go > 1 and yards_to_go <= 5: # Short (2-5 yards to go)
        return 1
    elif yards_to_go > 5 and yards_to_go <= 9: # Medium (6-9 yards to go)
        return 2
    else: # Long (10+ yards to go)
        return 3
    
def encode_passLength(pass_length):
    if pd.isna(pass_length):
        return -1
    
    if pass_length <= 7: # Short (0-7 yards)
        return 0
    elif pass_length > 7 and pass_length <= 15: # Medium (8-15 yards)
        return 1
    else: # Deep (16+ yards)
        return 2
    
def encode_offenseFormation(formation):
    if pd.isna(formation):
        return -1
    
    if formation == 'SHOTGUN':
        return 0
    elif formation == 'SINGLEBACK':
        return 1
    elif formation == 'EMPTY':
        return 2
    elif formation == 'I_FORM':
        return 3
    elif formation == 'PISTOL':
        return 4
    elif formation == 'JUMBO':
        return 5
    elif formation == 'WILDCAT':
        return 6
    
def encode_receiverAlignment(alignment):
    if pd.isna(alignment):
        return -1
    
    if alignment == '2x2':
        return 0
    elif alignment == '3x1':
        return 1
    elif alignment == '2x1':
        return 2
    elif alignment == '3x2':
        return 3
    elif alignment == '4x1':
        return 4
    elif alignment == '1x1':
        return 5
    elif alignment == '2x0':
        return 6
    
def encode_passCoverage(coverage):
    if pd.isna(coverage) or coverage == None:
        return -1
    
    if 'Cover-1' in coverage:
        return 0
    elif 'Cover-2' in coverage:
        return 1
    elif 'Cover-3' in coverage:
        return 2
    elif 'Cover-6' in coverage or 'Cover 6' in coverage:
        return 3
    elif coverage == 'Quarters':
        return 4
    elif coverage == 'Red Zone':
        return 5
    elif coverage == 'Cover-0':
        return 6
    elif coverage == '2-Man':
        return 7
    elif coverage == 'Prevent':
        return 8
    elif coverage == 'Goal Line':
        return 9
    elif coverage == 'Bracket':
        return 10
    elif coverage == 'Miscellaneous':
        return 11
    
def encode_manZone(val):
    if pd.isna(val):
        return -1
    
    if val == 'Man':
        return 0
    elif val == 'Zone':
        return 1
    else:
        return 2
    
def encode_targetY(targetY):
    # Middle     = area between the hash marks
    # Left hash  = 23.9 yards from sideline
    # Hash width = 6.2 yards

    if pd.isna(targetY):
        return -1

    if targetY <= 23.9:
        return 0
    elif targetY > 23.9 and targetY <= 29.4:
        return 1
    else:
        return 2
    
def encode_passResult(passResult):
    if pd.isna(passResult):
        return -1
    
    if passResult == 'C': # Complete
        return 0
    elif passResult == 'I': # Incomplete
        return 1
    elif passResult == 'S': # Sacked
        return 2
    elif passResult == 'IN': # Intercepted
        return 3
    elif passResult == 'R': # Scramble
        return 4
    
def get_possessionTeamScoreDiff(row):
    """
    H 21 - 13 V (Poss = H) | diff = 8      Winning
    H 21 - 13 V (Poss = V) | diff = 8      Losing
    H 21 - 31 V (Poss = H) | diff = -10    Losing
    H 21 - 31 V (Poss = V) | diff = -10    Winning
    """
    differential = row['preSnapHomeScore'] - row['preSnapVisitorScore']

    # Separate differential into possession margin
    if differential == 0:                           # Tie game
        possession_differential = 0
    elif differential <= 8:                         # 1 score game
        possession_differential = 1
    elif differential > 8 and differential <= 16:   # 2 score game
        possession_differential = 2
    elif differential > 16 and differential <= 24:  # 3 score game
        possession_differential = 3
    else:
        possession_differential = 4 # 4+ score game

    # Apply inverse for Visitor
    if row['visitorTeamAbbr'] == row['possessionTeam']:
        possession_differential *= -1

    return possession_differential

def encode_passLocation(row):
    if row['targetY'] == -1 or row['passLength'] == -1:
        return -1
    
    if row['targetY'] == 0 and row['passLength'] == 0:      # Short pass left
        return 0
    elif row['targetY'] == 0 and row['passLength'] == 1:    # Short pass middle
        return 1
    elif row['targetY'] == 0 and row['passLength'] == 2:    # Short pass middle
        return 2
    elif row['targetY'] == 1 and row['passLength'] == 0:    # Med pass left
        return 3
    elif row['targetY'] == 1 and row['passLength'] == 1:    # Med pass middle
        return 4
    elif row['targetY'] == 1 and row['passLength'] == 2:    # Med pass right
        return 5
    elif row['targetY'] == 2 and row['passLength'] == 0:    # Deep pass left
        return 6
    elif row['targetY'] == 2 and row['passLength'] == 1:    # Deep pass middle
        return 7
    elif row['targetY'] == 2 and row['passLength'] == 2:    # Short pass right
        return 8
    
def get_defensive_team(row):
    if row['possessionTeam'] == row['homeTeamAbbr']:
        return row['visitorTeamAbbr']
    else:
        return row['homeTeamAbbr']
    
def get_full_team_name(team_abbr):
    return {
        "ARI": "Arizona Cardinals",
        "ATL": "Atlanta Falcons",
        "BAL": "Baltimore Ravens",
        "BUF": "Buffalo Bills",
        "CAR": "Carolina Panthers",
        "CHI": "Chicago Bears",
        "CIN": "Cincinnati Bengals",
        "CLE": "Cleveland Browns",
        "DAL": "Dallas Cowboys",
        "DEN": "Denver Broncos",
        "DET": "Detroit Lions",
        "GB": "Green Bay Packers",
        "HOU": "Houston Texans",
        "IND": "Indianapolis Colts",
        "JAX": "Jacksonville Jaguars",
        "KC": "Kansas City Chiefs",
        "LV": "Las Vegas Raiders",
        "LAC": "Los Angeles Chargers",
        "LAR": "Los Angeles Rams",
        "MIA": "Miami Dolphins",
        "MIN": "Minnesota Vikings",
        "NE": "New England Patriots",
        "NO": "New Orleans Saints",
        "NYG": "New York Giants",
        "NYJ": "New York Jets",
        "PHI": "Philadelphia Eagles",
        "PIT": "Pittsburgh Steelers",
        "SF": "San Francisco 49ers",
        "SEA": "Seattle Seahawks",
        "TB": "Tampa Bay Buccaneers",
        "TEN": "Tennessee Titans",
        "WAS": "Washington Commanders",
    }.get(team_abbr, "Unknown Team")  # Default if abbreviation isn't found

team_abbr_to_name = {
    'ARI': 'Arizona Cardinals',
    'ATL': 'Atlanta Falcons',
    'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills',
    'CAR': 'Carolina Panthers',
    'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals',
    'CLE': 'Cleveland Browns',
    'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos',
    'DET': 'Detroit Lions',
    'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans',
    'IND': 'Indianapolis Colts',
    'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs',
    'LV': 'Las Vegas Raiders',
    'LAC': 'Los Angeles Chargers',
    'LA': 'Los Angeles Rams',
    'MIA': 'Miami Dolphins',
    'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots',
    'NO': 'New Orleans Saints',
    'NYG': 'New York Giants',
    'NYJ': 'New York Jets',
    'PHI': 'Philadelphia Eagles',
    'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers',
    'SEA': 'Seattle Seahawks',
    'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans',
    'WAS': 'Washington Commanders',
}
    