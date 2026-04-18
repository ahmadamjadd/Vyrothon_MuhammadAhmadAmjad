#!/usr/bin/env python3
"""Synthetic training data generator for Pocket-Agent."""
import json, random, os

random.seed(42)

SYSTEM_PROMPT = """You are a mobile assistant with these tools:

1. weather — location (string), unit ("C" or "F")
2. calendar — action ("list" or "create"), date ("YYYY-MM-DD"), title (string, optional for list)
3. convert — value (number), from_unit (string), to_unit (string)
4. currency — amount (number), from (ISO 4217 3-letter code), to (ISO 4217 3-letter code)
5. sql — query (string)

If a request matches a tool, reply ONLY with:
<tool_call>{"tool": "name", "args": {...}}</tool_call>

If no tool fits (chitchat, unsupported request, ambiguous with no history), reply with plain text. Never invent tools."""

CITIES = [
    "Tokyo","Paris","London","New York","Berlin","Sydney","Mumbai","Beijing",
    "Cairo","São Paulo","Moscow","Dubai","Singapore","Toronto","Los Angeles",
    "Chicago","San Francisco","Seattle","Boston","Miami","Dallas","Denver",
    "Amsterdam","Barcelona","Rome","Vienna","Prague","Dublin","Stockholm",
    "Oslo","Helsinki","Copenhagen","Warsaw","Bangkok","Seoul","Taipei",
    "Jakarta","Manila","Nairobi","Lagos","Mexico City","Buenos Aires",
    "Lima","Bogotá","Santiago","Riyadh","Doha","Karachi","Lahore","Delhi",
    "Kolkata","Dhaka","Hanoi","Kuala Lumpur","Cape Town","Zurich","Geneva",
    "Munich","Hamburg","Lyon","Marseille","Edinburgh","Manchester","Vancouver",
    "Montreal","Melbourne","Brisbane","Perth","Osaka","Kyoto","Shanghai",
    "Shenzhen","Istanbul","Athens","Lisbon","Madrid","Seville","Florence",
]

CURRENCIES = ["USD","EUR","GBP","JPY","INR","AUD","CAD","CHF","CNY","KRW",
              "BRL","MXN","SGD","HKD","NOK","SEK","DKK","NZD","ZAR","AED",
              "SAR","THB","TRY","RUB","PLN","PHP","IDR","MYR","EGP","PKR"]

UNIT_PAIRS = [
    ("km","mi"),("mi","km"),("m","ft"),("ft","m"),("cm","in"),("in","cm"),
    ("kg","lb"),("lb","kg"),("g","oz"),("oz","g"),("L","gal"),("gal","L"),
    ("mL","fl oz"),("fl oz","mL"),("C","F"),("F","C"),("km/h","mph"),
    ("mph","km/h"),("m/s","km/h"),("kg","g"),("lb","oz"),("cm","mm"),
    ("m","cm"),("km","m"),("yd","m"),("m","yd"),("mi","ft"),("acre","sqm"),
    ("sqft","sqm"),("sqm","sqft"),("cal","kJ"),("kJ","cal"),
]

CALENDAR_TITLES = [
    "Team Sync","Doctor appointment","Dentist visit","Lunch with Sarah",
    "Project review","Sprint planning","1:1 with manager","Client call",
    "Board meeting","Coffee chat","Interview","Code review","Demo day",
    "Standup","Retrospective","Design review","Budget meeting","Training",
    "Workshop","Offsite","Team dinner","Performance review","Brainstorm",
    "Product launch","Release planning","Town hall","Hackathon","Webinar",
    "Conference call","Status update","Strategy session","All hands",
]

SQL_TEMPLATES = [
    ("Show all users", "SELECT * FROM users"),
    ("Find users older than {n}", "SELECT * FROM users WHERE age > {n}"),
    ("Get total sales by region", "SELECT region, SUM(sales) AS total_sales FROM sales GROUP BY region"),
    ("Count orders per customer", "SELECT customer_id, COUNT(*) AS order_count FROM orders GROUP BY customer_id"),
    ("Find products priced above {n}", "SELECT * FROM products WHERE price > {n}"),
    ("Show employees in {dept} department", "SELECT * FROM employees WHERE department = '{dept}'"),
    ("Get average salary by department", "SELECT department, AVG(salary) AS avg_salary FROM employees GROUP BY department"),
    ("Find top {n} customers by revenue", "SELECT customer_id, SUM(amount) AS revenue FROM orders GROUP BY customer_id ORDER BY revenue DESC LIMIT {n}"),
    ("List all orders from {month} 2025", "SELECT * FROM orders WHERE strftime('%m', date) = '{month_num}' AND strftime('%Y', date) = '2025'"),
    ("Show inactive users", "SELECT * FROM users WHERE last_login < DATE('now', '-90 days')"),
    ("Find duplicate emails", "SELECT email, COUNT(*) AS cnt FROM users GROUP BY email HAVING cnt > 1"),
    ("Get monthly revenue for 2025", "SELECT strftime('%m', date) AS month, SUM(amount) AS revenue FROM orders WHERE strftime('%Y', date) = '2025' GROUP BY month"),
    ("Show all tables", "SELECT name FROM sqlite_master WHERE type='table'"),
    ("Count rows in users table", "SELECT COUNT(*) FROM users"),
    ("Find users who signed up this week", "SELECT * FROM users WHERE created_at >= DATE('now', '-7 days')"),
    ("Get the latest {n} orders", "SELECT * FROM orders ORDER BY date DESC LIMIT {n}"),
    ("Show all products in category {cat}", "SELECT * FROM products WHERE category = '{cat}'"),
    ("Find employees with salary above {n}", "SELECT * FROM employees WHERE salary > {n}"),
    ("Delete all expired sessions", "DELETE FROM sessions WHERE expires_at < DATETIME('now')"),
    ("Update user status to active", "UPDATE users SET status = 'active' WHERE id = {n}"),
]

DEPTS = ["Engineering","Marketing","Sales","HR","Finance","Legal","Support","Product","Design","Operations"]
CATS = ["Electronics","Clothing","Books","Food","Sports","Home","Garden","Toys","Beauty","Auto"]
MONTHS = {"January":"01","February":"02","March":"03","April":"04","May":"05","June":"06",
          "July":"07","August":"08","September":"09","October":"10","November":"11","December":"12"}

REFUSAL_PROMPTS = [
    "Tell me a joke","What's the meaning of life?","Who won the Super Bowl?",
    "Write me a poem","Send an email to John","Play some music",
    "Set an alarm for 7am","Take a photo","Call mom","Open the camera",
    "Turn on the flashlight","What's 2+2?","Translate hello to French",
    "Book a flight to Paris","Order pizza","Read my messages",
    "Set a timer for 5 minutes","What's the latest news?","How are you?",
    "Good morning!","Thanks!","Bye!","Who are you?","What can you do?",
    "Navigate to the nearest gas station","Find a restaurant nearby",
    "Remind me to buy milk","Track my package","Search for cat videos",
    "What time is it?","Open Settings","Restart my phone",
    "Post on Twitter","Send a text to Sarah","Take a screenshot",
    "Scan this QR code","Find my phone","Lock the door","Dim the lights",
    "Play the next song","What's trending?","Show my notifications",
    "Clear my cache","Uninstall this app","Rate this app",
    "How tall is Mount Everest?","What's the capital of France?",
    "Tell me about quantum physics","Explain machine learning",
    "What's the population of Earth?","When was the Eiffel Tower built?",
]

REFUSAL_RESPONSES = [
    "I'm sorry, I can't help with that. I can assist with weather, calendar, unit conversion, currency conversion, or database queries.",
    "That request doesn't match any of my available tools. I can help you with weather info, calendar management, unit conversions, currency conversions, or SQL queries.",
    "I don't have a tool for that. My capabilities include weather lookup, calendar management, unit conversion, currency conversion, and database queries. How can I help with one of those?",
    "That's outside my capabilities. I'm a tool-calling assistant that can help with weather, calendar, conversions, currency exchange, and database queries.",
    "I can't do that, but I can help you check the weather, manage your calendar, convert units or currencies, or run database queries.",
]

# -- Weather templates --
WEATHER_TEMPLATES = [
    ("What's the weather like in {city}?", "C"),
    ("Weather in {city}", "C"),
    ("How's the weather in {city}?", "C"),
    ("Tell me the weather for {city}", "C"),
    ("What's the temperature in {city}?", "C"),
    ("Is it hot in {city}?", "C"),
    ("Is it cold in {city} right now?", "C"),
    ("Give me the weather report for {city}", "C"),
    ("What's the forecast for {city}?", "C"),
    ("Check weather in {city}", "C"),
    ("Weather in {city} in Fahrenheit", "F"),
    ("What's the weather in {city} in Fahrenheit?", "F"),
    ("Tell me {city} temperature in Fahrenheit", "F"),
    ("How hot is it in {city} in F?", "F"),
    ("{city} weather in Fahrenheit please", "F"),
    ("Show me the weather in {city} in Celsius", "C"),
    ("What's it like outside in {city}?", "C"),
    ("Do I need an umbrella in {city}?", "C"),
    ("How warm is {city} today?", "C"),
    ("Current conditions in {city}?", "C"),
    ("Get me {city} weather in F", "F"),
    ("{city} forecast", "C"),
    ("What's the climate like in {city}?", "C"),
    ("Temperature check for {city}", "C"),
    ("How's {city} looking weather-wise?", "C"),
]

# -- Calendar templates --
CAL_LIST_TEMPLATES = [
    "What's on my calendar for {date}?",
    "Show my schedule for {date}",
    "What events do I have on {date}?",
    "List my calendar for {date}",
    "Any meetings on {date}?",
    "What do I have planned for {date}?",
    "Check my calendar on {date}",
    "Agenda for {date}?",
    "Do I have anything on {date}?",
    "What's happening on {date}?",
]
CAL_CREATE_TEMPLATES = [
    "Schedule {title} on {date}",
    "Create an event called {title} for {date}",
    "Add {title} to my calendar on {date}",
    "Book {title} on {date}",
    "Put {title} on {date}",
    "Set up {title} for {date}",
    "New event: {title} on {date}",
    "Add a calendar event {title} on {date}",
    "I need to schedule {title} on {date}",
    "Can you create {title} on {date}?",
    "Plan {title} for {date}",
    "Mark {date} for {title}",
]

# -- Convert templates --
CONVERT_TEMPLATES = [
    "Convert {val} {fu} to {tu}",
    "How many {tu} is {val} {fu}?",
    "What's {val} {fu} in {tu}?",
    "{val} {fu} to {tu}",
    "How much is {val} {fu} in {tu}?",
    "Change {val} {fu} to {tu}",
    "What does {val} {fu} equal in {tu}?",
    "I need to convert {val} {fu} to {tu}",
    "Please convert {val} {fu} into {tu}",
    "{val} {fu} equals how many {tu}?",
]

# -- Currency templates --
CURRENCY_TEMPLATES = [
    "Convert {amt} {fc} to {tc}",
    "How much is {amt} {fc} in {tc}?",
    "{amt} {fc} to {tc}",
    "What's {amt} {fc} in {tc}?",
    "Exchange {amt} {fc} to {tc}",
    "Change {amt} {fc} into {tc}",
    "How many {tc} is {amt} {fc}?",
    "I want to convert {amt} {fc} to {tc}",
    "Give me {amt} {fc} in {tc}",
    "{amt} {fc} equals how much in {tc}?",
]

# -- SQL templates --
SQL_QUERY_TEMPLATES = [
    "Show me {desc}",
    "{desc}",
    "Can you {desc}?",
    "I need to {desc}",
    "Run a query to {desc}",
    "Query: {desc}",
    "Please {desc}",
    "Help me {desc}",
]

# -- Adversarial: typos --
def add_typos(text, n_typos=1):
    chars = list(text)
    for _ in range(n_typos):
        if len(chars) < 3: break
        idx = random.randint(1, len(chars)-2)
        op = random.choice(["swap","drop","dup","replace"])
        if op == "swap" and idx < len(chars)-1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
        elif op == "drop":
            chars.pop(idx)
        elif op == "dup":
            chars.insert(idx, chars[idx])
        elif op == "replace":
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)

# -- Adversarial: code-switched prompts --
CODE_SWITCH_WEATHER = [
    ("मुझे {city} का weather बताओ", "C"),
    ("Kya {city} mein garmi hai?", "C"),
    ("{city} ka mausam kaisa hai?", "C"),
    ("Dime el clima en {city}", "C"),
    ("¿Cómo está el tiempo en {city}?", "C"),
    ("أريد معرفة الطقس في {city}", "C"),
    ("{city} mein weather kya hai Fahrenheit mein?", "F"),
    ("Mujhe {city} weather Celsius mein chahiye", "C"),
    ("{city} का तापमान बताइए", "C"),
    ("Dame el pronóstico de {city} en Fahrenheit", "F"),
]
CODE_SWITCH_CONVERT = [
    ("Convert karo {val} {fu} ko {tu} mein", None),
    ("{val} {fu} ko {tu} mein badlo", None),
    ("Convierte {val} {fu} a {tu}", None),
    ("حول {val} {fu} إلى {tu}", None),
    ("{val} {fu} se {tu} mein kitna hoga?", None),
]
CODE_SWITCH_CURRENCY = [
    ("{amt} {fc} ko {tc} mein convert karo", None),
    ("Mujhe {amt} {fc} {tc} mein chahiye", None),
    ("Convierte {amt} {fc} a {tc}", None),
    ("{amt} {fc} kitne {tc} hain?", None),
]

def make_tool_call(tool, args):
    return f'<tool_call>{json.dumps({"tool": tool, "args": args})}</tool_call>'

def rand_date():
    y = random.choice([2025, 2026])
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"

def generate_weather(n=300):
    examples = []
    for _ in range(n):
        city = random.choice(CITIES)
        tmpl, unit = random.choice(WEATHER_TEMPLATES)
        prompt = tmpl.format(city=city)
        resp = make_tool_call("weather", {"location": city, "unit": unit})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    return examples

def generate_calendar(n=250):
    examples = []
    for _ in range(n // 2):
        date = rand_date()
        tmpl = random.choice(CAL_LIST_TEMPLATES)
        prompt = tmpl.format(date=date)
        resp = make_tool_call("calendar", {"action": "list", "date": date})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    for _ in range(n // 2):
        date = rand_date()
        title = random.choice(CALENDAR_TITLES)
        tmpl = random.choice(CAL_CREATE_TEMPLATES)
        prompt = tmpl.format(date=date, title=title)
        resp = make_tool_call("calendar", {"action": "create", "date": date, "title": title})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    return examples

def generate_convert(n=250):
    examples = []
    for _ in range(n):
        fu, tu = random.choice(UNIT_PAIRS)
        val = round(random.uniform(0.5, 10000), random.choice([0,1,2]))
        if val == int(val): val = int(val)
        tmpl = random.choice(CONVERT_TEMPLATES)
        prompt = tmpl.format(val=val, fu=fu, tu=tu)
        resp = make_tool_call("convert", {"value": val, "from_unit": fu, "to_unit": tu})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    return examples

def generate_currency(n=200):
    examples = []
    for _ in range(n):
        fc, tc = random.sample(CURRENCIES, 2)
        amt = round(random.uniform(1, 100000), random.choice([0,2]))
        if amt == int(amt): amt = int(amt)
        tmpl = random.choice(CURRENCY_TEMPLATES)
        prompt = tmpl.format(amt=amt, fc=fc, tc=tc)
        resp = make_tool_call("currency", {"amount": amt, "from": fc, "to": tc})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    return examples

def generate_sql(n=200):
    examples = []
    for _ in range(n):
        desc_tmpl, query_tmpl = random.choice(SQL_TEMPLATES)
        n_val = random.choice([5, 10, 20, 25, 30, 50, 100, 500, 1000, 50000, 100000])
        dept = random.choice(DEPTS)
        cat = random.choice(CATS)
        month_name = random.choice(list(MONTHS.keys()))
        month_num = MONTHS[month_name]
        desc = desc_tmpl.format(n=n_val, dept=dept, cat=cat, month=month_name, month_num=month_num)
        query = query_tmpl.format(n=n_val, dept=dept, cat=cat, month=month_name, month_num=month_num)
        wrapper = random.choice(SQL_QUERY_TEMPLATES)
        prompt = wrapper.format(desc=desc.lower() if random.random() < 0.5 else desc)
        resp = make_tool_call("sql", {"query": query})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    return examples

def generate_refusals(n=200):
    examples = []
    prompts = REFUSAL_PROMPTS * ((n // len(REFUSAL_PROMPTS)) + 1)
    random.shuffle(prompts)
    for i in range(n):
        resp = random.choice(REFUSAL_RESPONSES)
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompts[i]},
            {"role": "assistant", "content": resp}
        ]})
    return examples

def generate_multiturn(n=150):
    examples = []
    # Currency follow-ups
    for _ in range(n // 3):
        fc, tc1, tc2 = random.sample(CURRENCIES, 3)
        amt = random.choice([50, 100, 200, 500, 1000, 5000])
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Convert {amt} {fc} to {tc1}"},
            {"role": "assistant", "content": make_tool_call("currency", {"amount": amt, "from": fc, "to": tc1})},
            {"role": "user", "content": random.choice([
                f"Now convert that to {tc2}",
                f"What about in {tc2}?",
                f"And in {tc2}?",
                f"How much would that be in {tc2}?",
                f"Same amount but in {tc2}",
            ])},
            {"role": "assistant", "content": make_tool_call("currency", {"amount": amt, "from": fc, "to": tc2})},
        ]
        examples.append({"messages": msgs})
    # Weather follow-ups
    for _ in range(n // 3):
        city = random.choice(CITIES)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"What's the weather in {city}?"},
            {"role": "assistant", "content": make_tool_call("weather", {"location": city, "unit": "C"})},
            {"role": "user", "content": random.choice([
                "What about in Fahrenheit?",
                "Show me that in F",
                "Convert that to Fahrenheit",
                "In Fahrenheit please",
                "And in Fahrenheit?",
            ])},
            {"role": "assistant", "content": make_tool_call("weather", {"location": city, "unit": "F"})},
        ]
        examples.append({"messages": msgs})
    # Convert follow-ups
    for _ in range(n // 3):
        fu, tu = random.choice(UNIT_PAIRS)
        val = random.choice([10, 25, 50, 100, 500])
        fu2, tu2 = random.choice(UNIT_PAIRS)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Convert {val} {fu} to {tu}"},
            {"role": "assistant", "content": make_tool_call("convert", {"value": val, "from_unit": fu, "to_unit": tu})},
            {"role": "user", "content": random.choice([
                f"Now convert {val*2} {fu2} to {tu2}",
                f"Also, what's {val*2} {fu2} in {tu2}?",
                f"And how about {val*2} {fu2} to {tu2}?",
            ])},
            {"role": "assistant", "content": make_tool_call("convert", {"value": val*2, "from_unit": fu2, "to_unit": tu2})},
        ]
        examples.append({"messages": msgs})
    return examples

def generate_adversarial(n=150):
    examples = []
    # Typo weather
    for _ in range(n // 5):
        city = random.choice(CITIES)
        tmpl, unit = random.choice(WEATHER_TEMPLATES[:10])
        prompt = add_typos(tmpl.format(city=city), random.randint(1,2))
        resp = make_tool_call("weather", {"location": city, "unit": unit})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    # Code-switched weather
    for _ in range(n // 5):
        city = random.choice(CITIES)
        tmpl, unit = random.choice(CODE_SWITCH_WEATHER)
        prompt = tmpl.format(city=city)
        resp = make_tool_call("weather", {"location": city, "unit": unit})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    # Code-switched convert
    for _ in range(n // 5):
        fu, tu = random.choice(UNIT_PAIRS)
        val = random.choice([10, 50, 100, 500])
        tmpl, _ = random.choice(CODE_SWITCH_CONVERT)
        prompt = tmpl.format(val=val, fu=fu, tu=tu)
        resp = make_tool_call("convert", {"value": val, "from_unit": fu, "to_unit": tu})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    # Code-switched currency
    for _ in range(n // 5):
        fc, tc = random.sample(CURRENCIES, 2)
        amt = random.choice([100, 500, 1000])
        tmpl, _ = random.choice(CODE_SWITCH_CURRENCY)
        prompt = tmpl.format(amt=amt, fc=fc, tc=tc)
        resp = make_tool_call("currency", {"amount": amt, "from": fc, "to": tc})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    # Typo convert/currency
    for _ in range(n // 5):
        fu, tu = random.choice(UNIT_PAIRS)
        val = random.choice([10, 25, 100])
        prompt = add_typos(f"Convert {val} {fu} to {tu}", 1)
        resp = make_tool_call("convert", {"value": val, "from_unit": fu, "to_unit": tu})
        examples.append({"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": resp}
        ]})
    return examples

def main():
    all_examples = []
    all_examples.extend(generate_weather(300))
    all_examples.extend(generate_calendar(250))
    all_examples.extend(generate_convert(250))
    all_examples.extend(generate_currency(200))
    all_examples.extend(generate_sql(200))
    all_examples.extend(generate_refusals(200))
    all_examples.extend(generate_multiturn(150))
    all_examples.extend(generate_adversarial(150))

    random.shuffle(all_examples)

    os.makedirs("data", exist_ok=True)
    with open("data/train.jsonl", "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(all_examples)} training examples → data/train.jsonl")

if __name__ == "__main__":
    main()
