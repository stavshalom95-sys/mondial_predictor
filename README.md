# Mondial Predictor

מנתח משחקי כדורגל (מונדיאל 2026) ומייעץ על ניחושים בטורניר חברים,
תוך שילוב מודל פואסון + תורת משחקים + מודעות לחוקי הניקוד.

פועל אוטומטית על GitHub Actions — ללא שרת חיצוני, בחינם.

---

## הקמה מאפס

### 1. צור repository
```bash
# במחשב המקומי
git init
git add .
git commit -m "init: mondial predictor"
```

צור repo **Private** ב-GitHub (יש מידע אישי בקובץ tournament_state.py),
ואז:
```bash
git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
git push -u origin master
```

### 2. הירשם לשירותים

**football-data.org** (לשליפת לוח משחקים):
- https://www.football-data.org/client/register
- קבל API key חינמי (Tier 0 = 10 קריאות/דקה — מספיק)

**Green-API** (לשליחת WhatsApp):
- https://green-api.com
- צור instance חינמי (Sandbox)
- סרוק QR מהטלפון האישי
- אשר את מספר היעד ב-Sandbox
- שמור את `instance_id` ואת ה-`token`

### 3. הוסף Secrets ב-GitHub

**Settings → Secrets and variables → Actions → New repository secret**

| שם ה-Secret | מה לשים |
|---|---|
| `FOOTBALL_DATA_API_KEY` | ה-API key מ-football-data.org |
| `GREEN_API_INSTANCE_ID` | מספר ה-instance מ-Green-API |
| `GREEN_API_TOKEN` | ה-token מ-Green-API |
| `WHATSAPP_RECIPIENT_PHONE` | מספר הטלפון היעד (פורמט: `972501234567`) |

### 4. הרצה ראשונה

עבור לטאב **Actions** ב-GitHub → בחר **Mondial Predictor — Daily Run** → לחץ **Run workflow**.
עקוב אחרי הלוג ווודא שהכל רץ תקין.

### 5. עדכון יומי (30 שניות)

ערוך את `get_todays_manual_odds()` ב-`main.py` עם יחסי הימורים של היום
(מכל אתר הימורים), ואז:

```bash
git add main.py
git commit -m "update odds: <תאריך>"
git push
```

ה-cron ירוץ אוטומטית ב-08:00 IDT (שעון קיץ).

---

## הרצה מקומית

```bash
# שליפת לוח משחקים (דורש FOOTBALL_DATA_API_KEY)
python scripts/fetch_schedule.py --output tests/sample_games.json

# הרצת הניתוח (ללא שליחה)
python main.py tests/sample_games.json --no-notify

# הרצה מלאה כולל שליחת WhatsApp
python main.py tests/sample_games.json
```

---

## מבנה הפרויקט

```
config/
  scoring_rules.py      # חוקי ניקוד + שוברי שוויון (מקור אמת יחיד)
core/
  odds_converter.py     # יחסי הימורים -> הסתברויות אמיתיות
  poisson_engine.py     # הסתברויות -> מטריצת תוצאות
  strategy_advisor.py   # שמרני/קונטרארי, מודע לשלב + שובר שוויון
data/
  data_pipeline.py      # פיענוח לוח משחקים מ-API
  tournament_state.py   # מצב נוכחי בטורניר (עדכון ידני בסיסי)
notifications/
  notifier.py           # עיצוב הודעת WhatsApp + שליחה דרך Green-API
scripts/
  fetch_schedule.py     # שליפת לוח משחקים מ-football-data.org
tests/
  sample_games.json     # נתוני דוגמה
.github/workflows/
  daily_run.yml         # GitHub Actions: cron יומי + הרצה ידנית
main.py                 # אורקסטרציה ראשית
project_spec.md         # מסמך מפרט (מקור אמת לפרויקט)
```

---

## הערות חשובות

- **יחסי הימורים לא נשלפים אוטומטית** — זוהי החלטה מכוונת (ר' §1 ב-project_spec.md).
  העדכון היומי של `get_todays_manual_odds()` הוא הפעולה היחידה שנדרשת ממך.
- **repo חייב להיות Private** — tournament_state.py מכיל מידע אישי.
- שעון קיץ: cron `0 5 * * *` = 08:00 IDT. בחורף (UTC+2) עבור ל-`0 6 * * *`.
