# Mondial Predictor — Project Spec

מסמך זה הוא מקור האמת לפרויקט. נכתב כדי ש-Claude Code (או כל מפתח/סוכן אחר)
יוכל להמשיך לעבוד על הפרויקט בלי לאבד הקשר מהשיחה המקורית שבה הוא תוכנן.

**תאריך עדכון:** 21 ביוני 2026
**שלב נוכחי:** שלב 1 הושלם (אסטרטגיית מרדף בטורניר חברים) — קוד נכתב ונבדק.
**שלב הבא:** הרצה אמיתית ב-GitHub + בדיקות פורמליות.

---

## 1. רקע ומטרה

כלי Python שמנתח משחקי כדורגל (מונדיאל 2026) ומייעץ אילו תוצאות לנחש
בטורניר ניחושים חברתי ("מונדיאל שטראוס", 10 משתתפים, 150 ש"ח כניסה),
תוך שילוב:

- **מודל פואסון** לחישוב הסתברויות לתוצאות מדויקות, מבוסס יחסי הימורים.
- **שכבת תורת משחקים** שממליצה מתי ללכת עם הקונצנזוס הסטטיסטי (שמרני)
  ומתי לבחור בתוצאה פחות צפויה כדי "לפתוח פער" על מתחרים (קונטרארי).
- **מודעות לחוקי הניקוד המדויקים** של הטורניר, כולל משקל שונה לכל שלב
  טורניר ולשוברי השוויון.
- **אוטומציה מלאה**: שליפת לוח משחקים, חישוב משחקים שנותרו, והתראת
  WhatsApp יומית — הכל רץ ב-GitHub Actions בחינם, ללא שרת חיצוני.

### החלטת היקף קריטית (אל תשנה בלי לדון מחדש)

המערכת **לא** מושכת יחסי הימורים אוטומטית מבתי הימורים מסחריים
(Bet365, "הווינר" וכו'), לא דרך scraping ולא דרך API. זו החלטה מכוונת:

- שליפת לוח משחקים/תוצאות: **אוטומטית**, ממקור ציבורי-חינמי (football-data.org).
- יחסי הימורים (odds): **קלט ידני יומי** בקובץ `main.py`, ~30 שניות עבודה.

הסיבה: המערכת נועדה להישאר "כלי ניתוח לטורניר חברים", לא תשתית
ל-scraping/מסחר אוטומטי בהימורים אמיתיים. אם בעתיד יישקל שינוי בכיוון
הזה (משיכת יחסים אוטומטית, ארביטראז' מול "הווינר" וכו'), זה דורש דיון
נפרד ומפורש — לא להניח שזו הרחבה טבעית של הקוד הקיים.

---

## 2. מבנה תיקיות מלא

```
mondial_predictor/
│
├── config/
│   ├── __init__.py
│   └── scoring_rules.py          # מקור אמת יחיד לחוקי ניקוד + שוברי שוויון
│
├── core/
│   ├── __init__.py
│   ├── odds_converter.py         # יחסי הימורים -> הסתברויות אמיתיות (הסרת overround)
│   ├── poisson_engine.py         # הסתברויות -> מטריצת תוצאות (Poisson 2D)
│   └── strategy_advisor.py       # שמרני/קונטרארי, מודע לשלב טורניר ולשובר שוויון
│
├── data/
│   ├── __init__.py
│   ├── data_pipeline.py          # פירוק לוח משחקים מ-API, חישוב "כמה משחקים נותרו"
│   └── tournament_state.py       # מצב נוכחי בטורניר (עדכון ידני בסיסי + fallback)
│
├── notifications/
│   ├── __init__.py
│   └── notifier.py               # עיצוב הודעת WhatsApp + שליחה דרך Green-API
│
├── scripts/
│   └── fetch_schedule.py         # שליפת לוח משחקים מ-football-data.org (ל-GitHub Actions)
│
├── tests/
│   └── sample_games.json         # נתוני דוגמה ל-pipeline (יש להוסיף pytest אמיתי - ר' §8)
│
├── .github/workflows/
│   └── daily_run.yml             # GitHub Actions: cron יומי 08:00 IDT + הרצה ידנית
│
├── main.py                       # אורקסטרציה: שליפה -> ניתוח -> התראה
└── README.md                     # הוראות הקמה מלאות (GitHub, Secrets, הרשמות API)
```

---

## 3. המצב שלי בטורניר (נתוני בסיס)

```python
my_points = 22
leader_points = 33          # doron gadesh
point_gap = 11

# הימורי טווח ארוך (12 נק' כל אחד אם נכון):
my_champion_pick = "Spain"
leader_champion_pick = "Spain"     # זהה - אין יתרון/חיסרון יחסי כאן
my_top_scorer_pick = "Harry Kane"
leader_top_scorer_pick = "Mbappe"  # היתרון האסימטרי שלי תלוי בקיין
```

קבועים אלה חיים ב-`data/tournament_state.py` (`MY_CURRENT_STATE`,
`LONG_TERM_BETS`). `matches_remaining` ב-`MY_CURRENT_STATE` הוא רק
ברירת מחדל — בפועל נדרס בכל הרצה ע"י `data_pipeline.matches_remaining_in_tournament()`.

---

## 4. חוקי הניקוד הרשמיים (Single Source of Truth: `config/scoring_rules.py`)

### 4.1 ניקוד טווח ארוך

| הימור | נקודות |
|---|---|
| נבחרת זוכה | 12 |
| מלך שערים | 12 (בשוויון: כל מי שבחר שחקן מהקבוצה הזוכה מקבל ניקוד מלא) |

### 4.2 ניקוד לפי שלב (כיוון נכון 1X2 לעומת תוצאה מדויקת/"בול")

| שלב | כיוון נכון | תוצאה מדויקת | מכפיל-ערך מול בתים (מחושב) |
|---|---|---|---|
| שלב הבתים | 1 | 3 | 1.0x |
| 32 האחרונות | 2 | 5 | 1.67x |
| שמינית גמר | 2 | 5 | 1.67x |
| רבע גמר | 4 | 8 | 2.67x |
| חצי גמר | 5 | 10 | 3.33x |
| מקום שלישי | 5 | 10 | 3.33x |
| הגמר הגדול | 8 | 15 | 5.0x |

המכפיל מחושב כ-`exact_score(stage) / exact_score(GROUP_STAGE)` ומשמש
את `strategy_advisor` לכיול הסף האדפטיבי (ר' §6).

### 4.3 שוברי שוויון (סדר עדיפות, מהחשוב לפחות חשוב)

1. **מספר התוצאות המדויקות ("בולים")** — הקריטי ביותר למודל הסיכון.
2. מספר הכיוונים הנכונים.
3. מספר השערים שכבש מלך השערים שנבחר.
4. ניחוש זהות הנבחרת המנצחת בטורניר (נכון/לא נכון).
5. זמן ההרשמה לטורניר (טאי-ברייקר אחרון, לא ניתן להשפעה).

**השלכה על המודל:** מאחר שספירת הבולים היא שובר השוויון הראשון, יש
ב-`strategy_advisor.py` קבוע `TIEBREAK_BOOST = 1.15` שמשקלל את חישובי
ה-EV לטובת תוצאות מדויקות (לא רק כיוון נכון) — תוצאה מדויקת משרתת שתי
מטרות בו-זמנית: ניקוד גולמי + מיקום טוב יותר בשובר השוויון.

---

## 5. ה-Pipeline המתמטי (odds → תוצאה מומלצת)

### שלב 1: `odds_converter.py` — הסרת מרווח בית ההימורים

יחסי הימורים גולמיים (Decimal Odds, למשל `home=1.65, draw=4.00, away=5.50`)
תמיד מסתכמים ליותר מ-100% הסתברות (overround/vig — הרווח של בית
ההימורים). לפני שימוש במודל, מנרמלים בשיטה הפרופורציונלית
(Multiplicative Method): כל הסתברות גולמית (`1/odds`) מחולקת בסכום
הכולל. גם לשוק Over/Under (קו שערים, בד"כ 2.5) יש פונקציית נרמול
מקבילה (`remove_overround_ou`).

### שלב 2: `poisson_engine.py` — כיול λ ובניית מטריצת תוצאות

1. מחפשים זוג `(lambda_home, lambda_away)` שמשחזר הכי טוב את ההסתברויות
   האמיתיות (1X2) שהתקבלו משלב 1, באמצעות **חיפוש גריד גס** (קפיצות
   0.1) ואז **עידון מקומי** (קפיצות 0.02) סביב הנקודה הטובה ביותר.
2. פרמטר `avg_total_goals_hint` (ברירת מחדל 2.6) מכוון את טווח החיפוש;
   אם יש נתוני Over/Under, מחשבים הערכה מדויקת יותר: `line + (p_over - p_under) * 1.2`.
3. בונים מטריצת `Poisson(lambda_home) × Poisson(lambda_away)` לכל
   `(home_goals, away_goals)` בין 0–8 (קבוע `MAX_GOALS = 8`), וממיינים
   מהסביר ביותר לפחות סביר.
4. תוצאה: `PoissonMatchModel` עם `top_n(n)` ו-`probability_of(h, a)`.

### שלב 3: `strategy_advisor.py` — ההחלטה התחרותית (הליבה הייחודית)

**העיקרון:** בטורניר חברים אתה לא מתחרה מול "המציאות הסטטיסטית" אלא
מול **ההתפלגות של ניחושי היריבים**. אם כולם ינחשו את אותה תוצאה
(הקונצנזוס), אף אחד לא סוגר פער גם אם היא קורית.

#### 5.1 זיהוי קונצנזוס

```python
def _is_likely_consensus_pick(scoreline, model, top_n_for_consensus=2) -> bool:
    # רק 2 התוצאות המובילות ביותר לפי פואסון נחשבות "קונצנזוס"
    # (לא רשימה קבועה כמו {1:0, 2:1, 1:1} - זו הייתה גרסה ראשונה שתוקנה,
    # כי היא חסמה כמעט את כל המועמדים הסבירים)
```

#### 5.2 איתור מועמד קונטררי

```python
def find_contrarian_candidate(model, min_probability=0.05, top_k_to_scan=20):
    # סורק את 20 התוצאות הסבירות ביותר, מחפש את הראשונה שאינה קונצנזוס
    # וגם לא "קלפי מדי" (מעל min_probability).
    # אם לא נמצא - מרכך את הסף בהדרגה (0.05 -> 0.03 -> 0.015) כ-fallback.
```

#### 5.3 הסף האדפטיבי-לשלב (החלק החדש/החשוב ביותר)

```python
multiplier = stage_value_multiplier(stage)          # למשל 5.0 בגמר
adjusted_threshold = point_gap_threshold / multiplier # בסיס: 2.0

if context.gap_per_match <= adjusted_threshold:
    strategy = SAFE
else:
    strategy = CONTRARIAN
```

**ההיגיון:** ככל שהשלב "שווה" יותר (נוקאאוט מאוחר), כך פחות פער-למשחק
נדרש כדי להצדיק קונטרארי — כי כל משחק שם שקול ליותר נקודות רגילות.
**זה נבדק ואומת**: עם `gap_per_match = 1.5` ו-`point_gap_threshold = 2.0`
הבסיסי:

| שלב | מכפיל | סף מתואם | תוצאה (gap=1.5) |
|---|---|---|---|
| שלב הבתים | 1.0x | 2.00 | שמרני |
| שמינית גמר | 1.67x | 1.20 | קונטרארי |
| רבע גמר | 2.67x | 0.75 | קונטרארי |
| חצי גמר | 3.33x | 0.60 | קונטרארי |
| גמר | 5.0x | 0.40 | קונטרארי |

#### 5.4 `TournamentContext` (קלט לכל ההחלטה)

```python
@dataclass
class TournamentContext:
    my_points: int
    leader_points: int
    matches_remaining: int          # נדרס אוטומטית ע"י data_pipeline בכל הרצה
    current_stage: TournamentStage = TournamentStage.GROUP_STAGE

    @property
    def point_gap(self) -> int: ...        # leader_points - my_points
    @property
    def gap_per_match(self) -> float: ...  # point_gap / matches_remaining
```

#### 5.5 פלט: `StrategyRecommendation`

כולל: `strategy` (SAFE/CONTRARIAN), `recommended_pick`, `reasoning`
(טקסט בעברית עם כל הנימוקים המספריים), `alternative_safe_pick`,
`expected_value_safe/contrarian`, `stage`, `points_if_exact`,
`points_if_direction_only`.

---

## 6. שכבת הנתונים האוטומטית (`data/data_pipeline.py`)

```python
@dataclass
class ScheduledMatch:
    match_id: str
    home_team: str
    away_team: str
    start_time_utc: datetime
    status: str  # "scheduled" | "live" | "final"
    home_score: int | None
    away_score: int | None

def parse_world_cup_schedule(raw_games: list[dict]) -> list[ScheduledMatch]: ...
def matches_remaining_in_tournament(all_matches, as_of=None) -> int: ...
def get_next_unplayed_matches(all_matches, limit=5) -> list[ScheduledMatch]: ...
def get_match_by_teams(all_matches, home_team, away_team) -> ScheduledMatch | None: ...
```

**פורמט הקלט הצפוי** (`raw_games`, רשימת dicts) — תואם הן לכלי הפנימי
`fetch_sports_data` (league="world_cup") והן לפלט של `scripts/fetch_schedule.py`:

```json
{
  "id": "sr:sport_event:66456998",
  "status": "scheduled",
  "start_time": "2026-06-21T16:00:00+00:00",
  "home": "ESP",
  "away": "KSA",
  "teams": {
    "ESP": {"name": "Spain", "abbreviation": "ESP"},
    "KSA": {"name": "Saudi Arabia", "abbreviation": "KSA"}
  },
  "score": {"ESP": 0, "KSA": 0}
}
```

נבדק מול ה-API החי (Sport Radar, דרך `fetch_sports_data`) ב-21.6.2026
— כולל משחקי 18-23 ביוני 2026 בפועל (קנדה-קטאר, ספרד-ערב הסעודית וכו').

---

## 7. שליפת לוח משחקים ב-GitHub Actions (`scripts/fetch_schedule.py`)

הסביבה הזו (claude.ai) משתמשת בכלי פנימי `fetch_sports_data`. ב-GitHub
Actions אין גישה לכלי הזה, אז `fetch_schedule.py` שולף מ-**football-data.org**
(API ציבורי, חינמי ב-Tier הבסיסי, 10 קריאות/דקה):

```python
url = f"https://api.football-data.org/v4/competitions/WC/matches"
# Header: X-Auth-Token: <FOOTBALL_DATA_API_KEY>
```

ממפה סטטוסים: `FINISHED→final`, `IN_PLAY/PAUSED→live`, `SCHEDULED/TIMED→scheduled`.
**נבדק ואומת** תאימות פורמט מלאה מול `parse_world_cup_schedule` (טסט
סינתטי עם תשובת API מדומה — עבר בהצלחה).

הרשמה: https://www.football-data.org/client/register

---

## 8. התראות WhatsApp (`notifications/notifier.py`)

### 8.1 ספק: Green-API (Sandbox חינמי)

נבחר על פני Twilio WhatsApp Business כי לא דורש אישור עסקי — רק סריקת
QR מהטלפון האישי. הרשמה: https://green-api.com

נדרשים 3 משתני סביבה (כ-GitHub Secrets):
- `GREEN_API_INSTANCE_ID`
- `GREEN_API_TOKEN`
- `WHATSAPP_RECIPIENT_PHONE` (פורמט בינלאומי, למשל `972501234567`)

### 8.2 שתי פונקציות מופרדות (טהור מול side-effect)

```python
def format_daily_message(picks: list[DailyPick], context: TournamentContext) -> str:
    # פונקציה טהורה - לבדוק עם assert בלי לשלוח כלום

def send_whatsapp_message(message, instance_id=None, api_token=None, recipient_phone=None) -> bool:
    # שכבת רשת. אם חסרים credentials -> מדפיס את ההודעה לטרמינל ומחזיר False
    # (לא קורס, כדי שכישלון שליחה לא יפיל את כל ה-pipeline)
```

### 8.3 פורמט ההודעה (דוגמה אמיתית מבדיקה)

```
⚽ *תחזית מונדיאל שטראוס - היום*
📊 מצב נוכחי: 22 נק' (אתה) | 33 נק' (מוביל)
📉 פער: 11 נק' | 2 משחקים נותרו

🎲 *Spain נגד Saudi Arabia*
   ניחוש: *1:0* (10% סיכוי)
   אסטרטגיה: קונטרארי | שלב: שלב הבתים
   ניקוד: בול=3 | כיוון=1
   (קונצנזוס היה: 2:0)

_נשלח אוטומטית ע"י Mondial Predictor_
```

האייקון `🛡️` לשמרני, `🎲` לקונטרארי.

---

## 9. אורקסטרציה (`main.py`)

```python
def get_todays_manual_odds() -> dict[str, dict]:
    """
    *הדבר היחיד שדורש עדכון ידני יומי.*
    מפתח: (home_team, away_team) -> {"odds_1x2": MatchOdds1X2, "ou_odds": OverUnderOdds, "stage": TournamentStage}
    """

def run_daily_pipeline(raw_games_from_api: list[dict], send_notification: bool = True) -> str:
    # 1. parse_world_cup_schedule + matches_remaining_in_tournament -> עדכון MY_CURRENT_STATE
    # 2. לכל משחק ב-get_todays_manual_odds(): match לפי get_match_by_teams,
    #    דילוג אם status == "final", הרצת odds_converter -> poisson_engine -> strategy_advisor
    # 3. format_daily_message + send_whatsapp_message
    # מחזיר את טקסט ההודעה (גם אם send_notification=False) - נוח לבדיקות
```

הרצה: `python main.py <path_to_games_json>` (הקובץ ייווצר אוטומטית
ע"י `fetch_schedule.py` בזרימת ה-CI).

---

## 10. GitHub Actions (`.github/workflows/daily_run.yml`)

```yaml
on:
  schedule:
    - cron: "0 5 * * *"   # UTC. 05:00 UTC = 08:00 IDT (קיץ, UTC+3).
                           # בחורף (UTC+2) זה ייצא 07:00 IDT - יש לכוונן ידנית בעונה.
  workflow_dispatch: {}    # מאפשר הרצה ידנית מטאב Actions, לבדיקות

jobs:
  run-predictor:
    runs-on: ubuntu-latest
    steps:
      - actions/checkout@v4
      - actions/setup-python@v5 (3.11)
      - pip install requests
      - python scripts/fetch_schedule.py --output tests/sample_games.json
        env: FOOTBALL_DATA_API_KEY (secret)
      - python main.py tests/sample_games.json
        env: GREEN_API_INSTANCE_ID, GREEN_API_TOKEN, WHATSAPP_RECIPIENT_PHONE (secrets)
```

**הקמה מאפס** (מפורט גם ב-`README.md`):
1. צור repo **Private** ב-GitHub (יש מידע אישי).
2. `git init && git add . && git commit -m "init" && git push`.
3. הירשם ל-football-data.org, קבל API key.
4. הירשם ל-Green-API, צור instance, סרוק QR, אשר את מספר היעד ב-Sandbox.
5. הוסף 4 Secrets ב-Settings → Secrets and variables → Actions.
6. הרץ ידנית פעם אחת דרך טאב Actions → Run workflow, ועקוב אחרי הלוג.
7. עדכון יומי: רק `get_todays_manual_odds()` ב-`main.py`, ואז `git push`.

---

## 11. מה נבדק בפועל (כל הבדיקות עברו)

- [x] `odds_converter`: overround מחושב נכון (יצא 3.79% על יחסים לדוגמה).
- [x] `poisson_engine`: λ מכויל נכון, מטריצת תוצאות ממוינת כראוי.
- [x] `strategy_advisor` v1→v2: תוקן באג שבו רשימת "תוצאות פופולריות"
      קבועה (`{1:0, 2:1, 1:1, 2:0, 0:0}`) חסמה כמעט כל מועמד קונטררי;
      הוחלף ב-top-N דינמי לפי המודל עצמו.
- [x] `strategy_advisor` v2: מוּכח שהמכפיל-לפי-שלב משנה את תוצאת
      ההחלטה בפועל (טבלה ב-§5.3), לא רק "נראה טוב על הנייר".
- [x] `data_pipeline.parse_world_cup_schedule`: נבדק מול תשובת API
      אמיתית של המונדיאל 2026 (משחקים מ-18–23 ביוני בפועל).
- [x] `fetch_schedule.py` ↔ `data_pipeline.py`: תאימות פורמט מלאה,
      נבדקה עם תשובת football-data.org מדומה.
- [x] `main.py` end-to-end: רץ מקצה לקצה (`tests/sample_games.json`),
      כולל fallback תקין של ה-notifier כשאין WhatsApp credentials.
- [x] כל המודולים נטענים ביחד ללא circular imports.

---

## 12. מה עוד לא בנוי / TODO ידוע

1. **עדכון אוטומטי של `current_stage`** — כרגע מוזן ידנית per-match
   בתוך `get_todays_manual_odds()`. אפשר לגזור אוטומטית משלב המונדיאל
   בפועל (יש את זה ב-metadata של ה-API, צריך מיפוי).
2. **טסטים פורמליים** — קיים רק `tests/sample_games.json` (קובץ נתונים).
   אין עדיין `pytest` עם assertions אמיתיים על odds_converter /
   poisson_engine / strategy_advisor. מומלץ להוסיף בהמשך.
3. **כיוון cron עונתי** — ה-`cron: "0 5 * * *"` מניח שעון קיץ ישראלי
   (UTC+3). בחורף יסיט את ההתראה לשעה 07:00 במקום 08:00 — לא קריטי
   לשלב 1 (מונדיאל הוא ביוני-יולי, בתוך הקיץ), אבל לתעד אם מורחב לעונה אחרת.
4. **שלב 2 (מנוע הימורי ערך מסחרי)** — נדון בתחילת הפרויקט ונדחה
   במפורש. ראו "החלטת היקף קריטית" ב-§1 — לא להניח שזו הרחבה טבעית.

---

## 13. קבצים — רשימה מלאה לבדיקת שלמות

```
config/__init__.py
config/scoring_rules.py
core/__init__.py
core/odds_converter.py
core/poisson_engine.py
core/strategy_advisor.py
data/__init__.py
data/data_pipeline.py
data/tournament_state.py
notifications/__init__.py
notifications/notifier.py
scripts/fetch_schedule.py
tests/sample_games.json
.github/workflows/daily_run.yml
main.py
README.md
project_spec.md   <- מסמך זה
```
