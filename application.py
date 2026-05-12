"""
The Math Stack: Student Meal Planner AI
========================================
A Streamlit application using Mixed-Integer Linear Programming (MILP) via PuLP
to optimize student nutrition on a budget.

Math in AI Usage:
  1. MILP (PuLP)         — Objective function maximizes protein, subject to budget + calorie constraints
  2. Harris-Benedict BMR — Calculates personalized calorie targets from biometric inputs
  3. Inventory Weighting — +100 bonus added to MILP objective coefficients for preferred ingredients
  4. Shannon Entropy      — Measures meal-plan diversity (Science tab explainer)
  5. Operations Research  — Full OR pipeline: model → solve → interpret (Science tab explainer)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pulp
import math

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="The Math Stack — Student Meal Planner AI",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (dark academic / retro-futuristic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Serif+Display:ital@0;1&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
  --bg:        #0d0f14;
  --surface:   #151821;
  --border:    #252a38;
  --accent:    #f5c518;
  --accent2:   #3af0b0;
  --muted:     #7a8099;
  --text:      #e8eaf2;
  --danger:    #ff5e5b;
  --card-bg:   #1a1e2b;
}

html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'IBM Plex Sans', sans-serif;
}

/* Headings */
h1, h2, h3 { font-family: 'DM Serif Display', serif; letter-spacing: .02em; }
h1 { font-size: 2.6rem; color: var(--accent); }
h2 { font-size: 1.8rem; color: var(--accent2); }
h3 { font-size: 1.2rem; color: var(--text); }

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] label { font-family: 'Space Mono', monospace; font-size:.75rem; color: var(--muted); letter-spacing:.08em; text-transform:uppercase; }

/* Metric cards */
.metric-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.4rem;
  text-align: center;
}
.metric-label { font-family:'Space Mono',monospace; font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; }
.metric-value { font-family:'DM Serif Display',serif; font-size:2rem; color:var(--accent); margin:.15rem 0; }
.metric-sub   { font-size:.75rem; color:var(--muted); }

/* Meal card */
.meal-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 1rem;
  transition: border-color .2s;
}
.meal-card:hover { border-color: var(--accent2); }
.meal-img { width:100%; height:155px; object-fit:cover; }
.meal-body { padding:.75rem 1rem; }
.meal-name { font-family:'DM Serif Display',serif; font-size:1.05rem; color:var(--accent); margin:0 0 .3rem; }
.meal-macro {
  display:flex; gap:.5rem; flex-wrap:wrap;
  font-family:'Space Mono',monospace; font-size:.65rem; color:var(--muted);
  margin-bottom:.5rem;
}
.pill { background:var(--border); border-radius:20px; padding:.15rem .55rem; }
.pill-green { background:#1a3028; color:var(--accent2); }
.pill-gold  { background:#2e2610; color:var(--accent); }
.pill-red   { background:#2e1515; color:var(--danger); }

/* Tabs */
[data-baseweb="tab"] { font-family:'Space Mono',monospace; font-size:.8rem; color:var(--muted); }
[aria-selected="true"] { color:var(--accent) !important; border-bottom:2px solid var(--accent) !important; }

/* Expander */
[data-testid="stExpander"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
}

/* Buttons */
.stButton > button {
  background: var(--border);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 6px;
  font-family:'Space Mono',monospace;
  font-size:.72rem;
  transition: background .15s, border-color .15s;
}
.stButton > button:hover {
  background: #1f2535;
  border-color: var(--accent2);
  color: var(--accent2);
}

/* Solve button */
.solve-btn > button {
  background: var(--accent) !important;
  color: #0d0f14 !important;
  font-weight: 700 !important;
  border: none !important;
  width: 100%;
  padding: .6rem;
  font-size: .9rem !important;
  border-radius: 8px !important;
}

/* Info box */
.info-box {
  background: #111827;
  border-left: 3px solid var(--accent2);
  padding: .8rem 1.1rem;
  border-radius: 0 8px 8px 0;
  font-size: .85rem;
  margin: .8rem 0;
}
.math-tag {
  display:inline-block;
  background:#0f1e33;
  border:1px solid #1a4a80;
  color:#6db3f2;
  font-family:'Space Mono',monospace;
  font-size:.65rem;
  padding:.1rem .45rem;
  border-radius:4px;
  margin:.15rem .1rem;
}

/* Divider */
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA  — 200 student-friendly meals
# ─────────────────────────────────────────────
@st.cache_data
def get_meal_data() -> pd.DataFrame:
    meals = [
        # ── BREAKFAST (IDs 0-69) ──────────────────────────────────────────────
        ("Peanut Butter Oat Bowl",          380, 14, 5.5,  "oats, peanut butter, banana, milk, honey",
         "1. Cook oats with milk for 5 min.\n2. Stir in peanut butter.\n3. Top with sliced banana and drizzle honey.",
         "https://images.unsplash.com/photo-1516714819001-8ee7a13b71d7?w=400&q=70"),
        ("Classic Scrambled Eggs",          310, 21, 4.0,  "eggs, butter, milk, salt, pepper",
         "1. Whisk 3 eggs with 2 tbsp milk.\n2. Melt butter in pan on low heat.\n3. Add eggs and stir slowly until just set.",
         "https://images.unsplash.com/photo-1510693206972-df098062cb71?w=400&q=70"),
        ("Avocado Toast",                   350, 9,  7.0,  "bread, avocado, lemon, salt, chili flakes",
         "1. Toast bread.\n2. Mash avocado with lemon juice.\n3. Spread, season with salt and chili flakes.",
         "https://images.unsplash.com/photo-1541519227354-08fa5d50c820?w=400&q=70"),
        ("Greek Yogurt Parfait",            290, 18, 6.5,  "greek yogurt, granola, berries, honey",
         "1. Layer yogurt in a bowl.\n2. Add granola.\n3. Top with mixed berries and honey.",
         "https://images.unsplash.com/photo-1488477181946-6428a0291777?w=400&q=70"),
        ("Banana Protein Smoothie",         330, 22, 5.0,  "banana, protein powder, milk, peanut butter, ice",
         "1. Add all ingredients to blender.\n2. Blend 45 seconds until smooth.\n3. Pour and serve immediately.",
         "https://images.unsplash.com/photo-1553530979-7ee52a2670c4?w=400&q=70"),
        ("Whole Wheat Pancakes",            410, 12, 5.5,  "whole wheat flour, eggs, milk, baking powder, honey",
         "1. Mix flour, baking powder, egg, and milk.\n2. Cook on greased pan 2 min per side.\n3. Drizzle with honey.",
         "https://images.unsplash.com/photo-1567620905732-2d1ec7ab7445?w=400&q=70"),
        ("Egg & Cheese Sandwich",           430, 23, 5.0,  "eggs, cheese, bread, butter, tomato",
         "1. Fry egg in butter.\n2. Layer egg and cheese on toast.\n3. Add tomato slices.",
         "https://images.unsplash.com/photo-1550507992-eb63ffee0847?w=400&q=70"),
        ("Overnight Chia Oats",             310, 11, 6.0,  "oats, chia seeds, milk, vanilla, mango",
         "1. Mix oats, chia seeds, and milk.\n2. Refrigerate overnight.\n3. Top with mango chunks.",
         "https://images.unsplash.com/photo-1495214783159-3503fd1b572d?w=400&q=70"),
        ("Almond Butter Toast",             360, 10, 6.5,  "bread, almond butter, banana, cinnamon",
         "1. Toast bread golden.\n2. Spread almond butter generously.\n3. Top with banana slices and cinnamon.",
         "https://images.unsplash.com/photo-1542759564-7ccbb6ac450a?w=400&q=70"),
        ("Veggie Omelette",                 290, 20, 5.5,  "eggs, bell pepper, onion, spinach, olive oil",
         "1. Sauté pepper and onion 3 min.\n2. Add spinach, cook 1 min.\n3. Pour beaten eggs, fold when set.",
         "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=400&q=70"),
        ("Cottage Cheese Bowl",             250, 24, 5.0,  "cottage cheese, pineapple, honey, walnuts",
         "1. Scoop cottage cheese into bowl.\n2. Add pineapple pieces.\n3. Drizzle honey and scatter walnuts.",
         "https://images.unsplash.com/photo-1559181567-c3190ca9be46?w=400&q=70"),
        ("Fruit & Nut Granola Bowl",        370, 9,  6.0,  "granola, milk, apple, raisins, almonds",
         "1. Pour granola into bowl.\n2. Add cold milk.\n3. Top with diced apple, raisins and almonds.",
         "https://images.unsplash.com/photo-1517093157656-b9eccef91cb1?w=400&q=70"),
        ("Shakshuka",                       360, 18, 6.5,  "eggs, tomatoes, onion, garlic, cumin, bell pepper",
         "1. Sauté onion, garlic, pepper 5 min.\n2. Add tomatoes and cumin, simmer 8 min.\n3. Crack eggs into sauce, cover until whites set.",
         "https://images.unsplash.com/photo-1590412200988-a436970781fa?w=400&q=70"),
        ("Hummus & Veggie Wrap",            390, 13, 6.0,  "tortilla, hummus, cucumber, tomato, spinach, carrot",
         "1. Spread hummus on tortilla.\n2. Layer veggies.\n3. Roll tightly and slice.",
         "https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=400&q=70"),
        ("Multigrain Cereal & Milk",        280, 8,  4.0,  "multigrain cereal, milk, strawberries",
         "1. Pour cereal into bowl.\n2. Add cold milk.\n3. Top with sliced strawberries.",
         "https://images.unsplash.com/photo-1521483451569-e33803c0330c?w=400&q=70"),
        ("Feta Egg Scramble",               320, 22, 6.5,  "eggs, feta cheese, tomato, spinach, olive oil",
         "1. Sauté spinach and tomato.\n2. Add beaten eggs.\n3. Crumble feta on top, scramble until set.",
         "https://images.unsplash.com/photo-1595295333158-4742f28fbd85?w=400&q=70"),
        ("Protein Waffle",                  400, 28, 7.0,  "protein powder, oats, egg, milk, vanilla",
         "1. Blend all ingredients into smooth batter.\n2. Pour into waffle iron.\n3. Cook until golden, top with honey.",
         "https://images.unsplash.com/photo-1551892374-ecf8754cf8b0?w=400&q=70"),
        ("Date & Walnut Oatmeal",           360, 10, 5.5,  "oats, dates, walnuts, milk, cinnamon",
         "1. Cook oats with milk.\n2. Stir in chopped dates.\n3. Top with walnuts and cinnamon.",
         "https://images.unsplash.com/photo-1516714819001-8ee7a13b71d7?w=400&q=70"),
        ("Smashed Egg on Rice",             350, 20, 4.5,  "rice, eggs, soy sauce, sesame oil, spring onion",
         "1. Cook rice.\n2. Fry egg sunny-side up.\n3. Place on rice, add soy sauce and sesame oil.",
         "https://images.unsplash.com/photo-1490645935967-10de6ba17061?w=400&q=70"),
        ("Mango Lassi & Toast",             300, 8,  5.0,  "mango, yogurt, milk, cardamom, bread",
         "1. Blend mango, yogurt, milk, cardamom.\n2. Toast bread.\n3. Serve lassi alongside toast.",
         "https://images.unsplash.com/photo-1610478920693-49be16e8b5fa?w=400&q=70"),
        # ── LUNCH (IDs 20-109) ───────────────────────────────────────────────
        ("Chicken Rice Bowl",               550, 42, 9.0,  "chicken breast, rice, broccoli, soy sauce, garlic",
         "1. Season and grill chicken 7 min per side.\n2. Cook rice.\n3. Steam broccoli, assemble with soy-garlic sauce.",
         "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400&q=70"),
        ("Lentil Soup",                     380, 20, 5.5,  "red lentils, onion, tomato, cumin, turmeric, garlic",
         "1. Sauté onion, garlic.\n2. Add lentils, tomato, spices and water.\n3. Simmer 20 min and blend partially.",
         "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400&q=70"),
        ("Tuna Salad Wrap",                 430, 32, 7.0,  "canned tuna, tortilla, lettuce, tomato, mayo, lemon",
         "1. Mix tuna with mayo and lemon.\n2. Layer on tortilla with lettuce and tomato.\n3. Roll firmly.",
         "https://images.unsplash.com/photo-1540420773420-3366772f4999?w=400&q=70"),
        ("Falafel Pita",                    480, 16, 7.5,  "falafel, pita, tahini, lettuce, tomato, cucumber",
         "1. Warm falafel in pan.\n2. Open pita, spread tahini.\n3. Stuff with falafel and veggies.",
         "https://images.unsplash.com/photo-1529006557810-274b9b2fc783?w=400&q=70"),
        ("Pasta Primavera",                 520, 18, 8.0,  "pasta, zucchini, bell pepper, tomatoes, parmesan, olive oil",
         "1. Boil pasta al dente.\n2. Sauté veggies in olive oil.\n3. Toss together with parmesan.",
         "https://images.unsplash.com/photo-1551892374-ecf8754cf8b0?w=400&q=70"),
        ("Black Bean Burrito",              570, 22, 7.5,  "black beans, tortilla, rice, salsa, cheese, sour cream",
         "1. Heat black beans with cumin.\n2. Warm tortilla.\n3. Fill with rice, beans, salsa, cheese and sour cream.",
         "https://images.unsplash.com/photo-1534352956036-cd81e27dd615?w=400&q=70"),
        ("Quinoa Veggie Bowl",              430, 15, 8.5,  "quinoa, chickpeas, spinach, feta, lemon, olive oil",
         "1. Cook quinoa.\n2. Roast chickpeas 20 min.\n3. Assemble with spinach, feta, lemon-olive oil dressing.",
         "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400&q=70"),
        ("Chicken Caesar Salad",            460, 38, 9.5,  "chicken breast, romaine, caesar dressing, croutons, parmesan",
         "1. Grill and slice chicken.\n2. Toss romaine with dressing.\n3. Add croutons, parmesan, and chicken.",
         "https://images.unsplash.com/photo-1546793665-c74683f339c1?w=400&q=70"),
        ("Egg Fried Rice",                  490, 16, 5.5,  "rice, eggs, soy sauce, peas, carrot, garlic, sesame oil",
         "1. Cook rice and cool.\n2. Scramble eggs, add veggies.\n3. Add rice, soy sauce, sesame oil.",
         "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=400&q=70"),
        ("Spicy Chickpea Curry",            440, 16, 6.0,  "chickpeas, tomatoes, onion, garam masala, turmeric, ginger",
         "1. Sauté onion, ginger.\n2. Add spices, tomatoes and chickpeas.\n3. Simmer 15 min.",
         "https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=400&q=70"),
        ("Grilled Cheese Tomato Soup",      520, 14, 7.0,  "cheese, bread, butter, tomatoes, onion, basil",
         "1. Make tomato soup: blend cooked tomatoes with basil.\n2. Butter bread, add cheese, grill both sides.",
         "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400&q=70"),
        ("Beef & Veggie Stir Fry",          580, 38, 11.0, "beef strips, broccoli, bell pepper, soy sauce, garlic, rice",
         "1. Marinate beef in soy and garlic.\n2. Stir-fry beef then veggies on high heat.\n3. Serve over rice.",
         "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400&q=70"),
        ("Sweet Potato & Lentil Soup",      360, 14, 5.5,  "sweet potato, lentils, onion, cumin, coriander, coconut milk",
         "1. Cook onion, add spices.\n2. Add sweet potato and lentils with water.\n3. Simmer 25 min, stir in coconut milk.",
         "https://images.unsplash.com/photo-1547592166-23ac45744acd?w=400&q=70"),
        ("Turkey Lettuce Wraps",            330, 30, 8.5,  "ground turkey, lettuce, hoisin sauce, garlic, ginger, onion",
         "1. Brown turkey with garlic and ginger.\n2. Add hoisin sauce.\n3. Spoon into lettuce leaves.",
         "https://images.unsplash.com/photo-1540420773420-3366772f4999?w=400&q=70"),
        ("Caprese Pasta",                   490, 16, 8.0,  "pasta, mozzarella, tomato, basil, olive oil, balsamic",
         "1. Cook pasta.\n2. Slice mozzarella and tomato.\n3. Toss all with olive oil and balsamic.",
         "https://images.unsplash.com/photo-1551892374-ecf8754cf8b0?w=400&q=70"),
        ("Noodle Veggie Broth",             310, 10, 5.0,  "noodles, vegetable broth, bok choy, mushroom, soy sauce, tofu",
         "1. Heat broth.\n2. Add noodles and tofu, cook 5 min.\n3. Add bok choy and mushroom, simmer 3 min.",
         "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=400&q=70"),
        ("Shrimp Tacos",                    480, 28, 12.0, "shrimp, corn tortillas, cabbage, lime, salsa, avocado",
         "1. Season and grill shrimp.\n2. Warm tortillas.\n3. Assemble with cabbage, salsa and avocado.",
         "https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=400&q=70"),
        ("Masoor Dal with Rice",            430, 18, 5.0,  "red lentils, rice, cumin, turmeric, onion, tomato",
         "1. Cook lentils with spices.\n2. Temper cumin in oil, pour over.\n3. Serve with steamed rice.",
         "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=400&q=70"),
        ("BLT Sandwich",                    470, 20, 8.0,  "bread, bacon, lettuce, tomato, mayo",
         "1. Grill bacon until crispy.\n2. Toast bread, spread mayo.\n3. Layer bacon, lettuce, tomato.",
         "https://images.unsplash.com/photo-1550507992-eb63ffee0847?w=400&q=70"),
        ("Cheese Quesadilla",               450, 18, 6.5,  "tortilla, cheese, salsa, sour cream, jalapeño",
         "1. Grate cheese onto half the tortilla.\n2. Add jalapeño, fold over.\n3. Cook on pan until golden both sides.",
         "https://images.unsplash.com/photo-1534352956036-cd81e27dd615?w=400&q=70"),
        # ── DINNER (IDs 40-199) ──────────────────────────────────────────────
        ("Grilled Salmon & Quinoa",         580, 46, 18.0, "salmon fillet, quinoa, lemon, dill, olive oil, asparagus",
         "1. Season salmon, grill 4 min per side.\n2. Cook quinoa.\n3. Roast asparagus with olive oil, serve together.",
         "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400&q=70"),
        ("Spaghetti Bolognese",             620, 34, 10.5, "spaghetti, ground beef, tomatoes, onion, garlic, basil",
         "1. Brown beef with onion and garlic.\n2. Add tomatoes and basil, simmer 20 min.\n3. Serve over al-dente spaghetti.",
         "https://images.unsplash.com/photo-1555949258-eb67b1ef0ceb?w=400&q=70"),
        ("Butter Chicken with Naan",        650, 40, 13.0, "chicken, tomato, butter, cream, garam masala, garlic, naan",
         "1. Marinate and grill chicken.\n2. Make sauce with butter, tomato, cream and spices.\n3. Add chicken, serve with warm naan.",
         "https://images.unsplash.com/photo-1565557623262-b51c2513a641?w=400&q=70"),
        ("Veggie Stir-Fry Noodles",         460, 14, 7.5,  "noodles, broccoli, carrot, bell pepper, soy sauce, sesame oil",
         "1. Cook noodles, drain.\n2. Stir-fry veggies on high heat.\n3. Add noodles and sauce, toss well.",
         "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=400&q=70"),
        ("Baked Chicken Thighs",            520, 42, 10.0, "chicken thighs, garlic, paprika, olive oil, lemon",
         "1. Mix garlic, paprika, olive oil and lemon.\n2. Coat chicken, refrigerate 30 min.\n3. Bake at 200°C for 35 min.",
         "https://images.unsplash.com/photo-1532550907401-a500c9a57435?w=400&q=70"),
        ("Veggie Biryani",                  540, 14, 7.0,  "basmati rice, mixed vegetables, biryani spices, onion, yogurt",
         "1. Fry onions until golden.\n2. Cook veggies with biryani spices.\n3. Layer with par-cooked rice, cook on dum 20 min.",
         "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=400&q=70"),
        ("Stuffed Bell Peppers",            450, 28, 9.5,  "bell peppers, ground beef, rice, tomatoes, cheese, onion",
         "1. Hollow peppers, pre-bake 10 min.\n2. Mix beef, rice, tomato, onion.\n3. Stuff peppers, top with cheese, bake 25 min.",
         "https://images.unsplash.com/photo-1518779578993-ec3579fee39f?w=400&q=70"),
        ("Tuna Pasta Bake",                 530, 32, 9.0,  "pasta, canned tuna, cheese, tomato sauce, mushrooms",
         "1. Mix cooked pasta with tuna and tomato sauce.\n2. Add mushrooms, top with cheese.\n3. Bake at 190°C for 20 min.",
         "https://images.unsplash.com/photo-1551892374-ecf8754cf8b0?w=400&q=70"),
        ("Dahl & Roti",                     430, 17, 5.0,  "lentils, roti, onion, tomato, cumin, coriander",
         "1. Cook lentils with spices.\n2. Temper mustard seeds and onion.\n3. Serve hot with warm roti.",
         "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=400&q=70"),
        ("Prawn Fried Rice",                510, 28, 12.0, "prawns, rice, eggs, soy sauce, spring onion, garlic",
         "1. Cook and cool rice.\n2. Stir-fry prawns with garlic.\n3. Add rice, eggs, soy sauce, toss on high heat.",
         "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=400&q=70"),
        ("Mushroom Risotto",                510, 14, 9.0,  "arborio rice, mushrooms, parmesan, white wine, onion, broth",
         "1. Sauté onion and mushrooms.\n2. Add rice and wine, stir until absorbed.\n3. Add broth ladle by ladle, finish with parmesan.",
         "https://images.unsplash.com/photo-1476124369491-e7addf5db371?w=400&q=70"),
        # FILLER MEALS 51-199 (generated systematically for variety)
        *[
            (f"Spiced Chicken Bowl #{i}",   500+i%80, 35+i%10, 8.0+(i%6)*.5,
             "chicken breast, rice, olive oil, garlic, lemon, cumin",
             "1. Season chicken.\n2. Grill 7 min per side.\n3. Serve over rice with lemon.",
             "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400&q=70")
            for i in range(50, 65)
        ],
        *[
            (f"Lentil & Veggie Curry #{i}", 400+i%70, 17+i%8,  5.5+(i%5)*.4,
             "lentils, tomato, onion, turmeric, cumin, coriander, ginger",
             "1. Sauté aromatics.\n2. Add lentils and tomato.\n3. Simmer 20 min with spices.",
             "https://images.unsplash.com/photo-1455619452474-d2be8b1e70cd?w=400&q=70")
            for i in range(65, 80)
        ],
        *[
            (f"Egg & Rice Stir Fry #{i}",   420+i%60, 18+i%9,  4.5+(i%5)*.3,
             "eggs, rice, soy sauce, garlic, sesame oil, peas",
             "1. Scramble eggs.\n2. Add cold rice and peas.\n3. Season with soy sauce and sesame.",
             "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=400&q=70")
            for i in range(80, 95)
        ],
        *[
            (f"Tuna Pasta #{i}",            490+i%50, 28+i%8,  7.5+(i%5)*.5,
             "pasta, canned tuna, tomato sauce, cheese, garlic",
             "1. Cook pasta.\n2. Mix tuna with tomato sauce.\n3. Combine and top with cheese.",
             "https://images.unsplash.com/photo-1551892374-ecf8754cf8b0?w=400&q=70")
            for i in range(95, 110)
        ],
        *[
            (f"Veggie Noodle Bowl #{i}",    380+i%60, 12+i%7,  5.0+(i%5)*.3,
             "noodles, bok choy, mushroom, soy sauce, ginger, sesame oil",
             "1. Boil noodles.\n2. Stir fry veggies.\n3. Toss together with sauce.",
             "https://images.unsplash.com/photo-1569718212165-3a8278d5f624?w=400&q=70")
            for i in range(110, 125)
        ],
        *[
            (f"Chickpea Salad #{i}",        350+i%50, 14+i%6,  5.0+(i%5)*.4,
             "chickpeas, cucumber, tomato, olive oil, lemon, parsley",
             "1. Drain and rinse chickpeas.\n2. Chop veggies.\n3. Toss with olive oil and lemon.",
             "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400&q=70")
            for i in range(125, 140)
        ],
        *[
            (f"Beef Rice Bowl #{i}",        560+i%70, 38+i%10, 10.0+(i%6)*.5,
             "beef strips, rice, soy sauce, garlic, broccoli",
             "1. Stir-fry beef.\n2. Add broccoli and sauce.\n3. Serve over rice.",
             "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400&q=70")
            for i in range(140, 155)
        ],
        *[
            (f"Greek Salad Wrap #{i}",      370+i%50, 12+i%6,  6.5+(i%5)*.3,
             "tortilla, feta, cucumber, tomato, olives, lettuce",
             "1. Chop salad ingredients.\n2. Mix with feta and olives.\n3. Wrap in tortilla.",
             "https://images.unsplash.com/photo-1565299585323-38d6b0865b47?w=400&q=70")
            for i in range(155, 170)
        ],
        *[
            (f"Salmon & Rice #{i}",         570+i%60, 42+i%8,  16.0+(i%6)*1.0,
             "salmon fillet, rice, soy sauce, lemon, ginger",
             "1. Pan-fry salmon 4 min per side.\n2. Cook rice.\n3. Serve with soy-ginger sauce.",
             "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400&q=70")
            for i in range(170, 185)
        ],
        *[
            (f"Cottage Cheese Oat Bowl #{i}", 310+i%40, 22+i%8, 5.0+(i%5)*.3,
             "cottage cheese, oats, honey, banana, cinnamon",
             "1. Cook oats.\n2. Stir in cottage cheese.\n3. Top with banana, honey and cinnamon.",
             "https://images.unsplash.com/photo-1516714819001-8ee7a13b71d7?w=400&q=70")
            for i in range(185, 200)
        ],
    ]
    # Trim/pad to exactly 200
    meals = meals[:200]
    cols = ["Name", "Calories", "Protein", "Cost", "Ingredients", "Recipe", "Image"]
    return pd.DataFrame(meals, columns=cols)


# ─────────────────────────────────────────────
# HARRIS-BENEDICT BMR  ← Math in AI #1
# ─────────────────────────────────────────────
def calc_target_calories(age: int, gender: str, weight_kg: float,
                          height_cm: float, activity: str) -> float:
    """
    Math in AI — Harris-Benedict Revised Equation for BMR,
    multiplied by TDEE activity factor.
    """
    if gender == "Male":
        bmr = 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age
    else:
        bmr = 447.593 + 9.247 * weight_kg + 3.098 * height_cm - 4.330 * age

    factors = {
        "Sedentary (little/no exercise)":  1.2,
        "Lightly active (1-3 days/week)":  1.375,
        "Moderately active (3-5 days/wk)": 1.55,
        "Very active (6-7 days/wk)":       1.725,
        "Athlete (2x/day training)":       1.9,
    }
    return bmr * factors.get(activity, 1.55)


# ─────────────────────────────────────────────
# MILP OPTIMIZER  ← Math in AI #2 + #3
# ─────────────────────────────────────────────
def solve_meal_plan(df: pd.DataFrame,
                    weekly_budget: float,
                    target_calories_per_day: float,
                    inventory_list: list[str]) -> tuple[list[int] | None, str]:
    """
    Math in AI — Mixed-Integer Linear Program (PuLP / CBC):

      Objective  : Maximize Σ (protein_i + inventory_bonus_i) · x_i
      Subject to :
        Σ cost_i · x_i          ≤ weekly_budget           [budget]
        Σ calories_i · x_i      ≥ 0.9 · target_weekly_kcal [calorie floor]
        Σ x_i                   = 21                       [exactly 21 meals]
        x_i ∈ {0, 1, 2, 3}                                 [integer, max 3 repeats]

    Inventory Weighting (Math in AI #3):
      If meal i contains any ingredient in inventory_list → +100 bonus to objective
    """
    n = len(df)
    target_weekly = target_calories_per_day * 7

    # Build inventory bonus vector
    inv_lower = [s.strip().lower() for s in inventory_list]
    bonus = np.zeros(n)
    for i, row in df.iterrows():
        ing = [x.strip().lower() for x in row["Ingredients"].split(",")]
        if any(item in ing for item in inv_lower):
            bonus[i] = 100.0

    prob = pulp.LpProblem("MealPlan", pulp.LpMaximize)

    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=3, cat="Integer") for i in range(n)]

    # Objective
    prob += pulp.lpSum((df.loc[i, "Protein"] + bonus[i]) * x[i] for i in range(n))

    # Budget constraint
    prob += pulp.lpSum(df.loc[i, "Cost"] * x[i] for i in range(n)) <= weekly_budget

    # Calorie floor (≥90% of target)
    prob += pulp.lpSum(df.loc[i, "Calories"] * x[i] for i in range(n)) >= 0.9 * target_weekly

    # Exactly 21 meals
    prob += pulp.lpSum(x[i] for i in range(n)) == 21

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, pulp.LpStatus[prob.status]

    plan = []
    for i in range(n):
        val = int(round(pulp.value(x[i])))
        plan.extend([i] * val)

    return plan[:21], "Optimal"


# ─────────────────────────────────────────────
# SHANNON ENTROPY  ← Math in AI #4
# ─────────────────────────────────────────────
def shannon_entropy(plan_indices: list[int]) -> float:
    """H = -Σ p_i · log2(p_i)  over unique meals in the plan."""
    from collections import Counter
    counts = Counter(plan_indices)
    total  = len(plan_indices)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧮 The Math Stack")
    st.markdown("<span style='font-family:Space Mono;font-size:.7rem;color:#7a8099;'>STUDENT MEAL PLANNER AI</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 👤 Personal Parameters")

    age     = st.slider("Age", 14, 22, 18)
    gender  = st.selectbox("Gender", ["Male", "Female"])
    weight  = st.slider("Weight (kg)", 40, 110, 65)
    height  = st.slider("Height (cm)", 140, 200, 168)
    activity = st.selectbox("Activity Level", [
        "Sedentary (little/no exercise)",
        "Lightly active (1-3 days/week)",
        "Moderately active (3-5 days/wk)",
        "Very active (6-7 days/wk)",
        "Athlete (2x/day training)",
    ], index=1)

    st.divider()
    st.markdown("### 💰 Weekly Budget (AED)")
    budget = st.slider("", 80, 400, 180, step=5)

    st.divider()
    st.markdown("### 🥫 Inventory (items you already have)")
    inventory_raw = st.text_area(
        "One ingredient per line:",
        placeholder="e.g.\neggs\nrice\nchicken breast\noats",
        height=110,
    )
    inventory_list = [s.strip() for s in inventory_raw.strip().splitlines() if s.strip()]

    st.divider()
    st.markdown(
        "<div class='solve-btn'>",
        unsafe_allow_html=True,
    )
    solve_btn = st.button("⚡  Solve My Week", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = get_meal_data()
daily_kcal = calc_target_calories(age, gender, weight, height, activity)
weekly_kcal = daily_kcal * 7


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<h1>🧮 The Math Stack</h1>
<p style='font-family:Space Mono;font-size:.8rem;color:#7a8099;letter-spacing:.12em;'>
STUDENT MEAL PLANNER AI &nbsp;·&nbsp; MIXED-INTEGER LINEAR PROGRAMMING &nbsp;·&nbsp; AMBASSADOR SCHOOL MATHLETICS 2026
</p>
""", unsafe_allow_html=True)

# Metric bar
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, val, sub in [
    (c1, "Daily Calories", f"{daily_kcal:,.0f}", "kcal target"),
    (c2, "Weekly Budget",  f"AED {budget}",      "hard limit"),
    (c3, "Total Meals",    "21",                  "3/day × 7 days"),
    (c4, "Calorie Floor",  f"{daily_kcal*7*.9:,.0f}", "90% of weekly"),
    (c5, "Inventory",      str(len(inventory_list)), "items loaded"),
]:
    col.markdown(f"""
    <div class='metric-card'>
      <div class='metric-label'>{label}</div>
      <div class='metric-value'>{val}</div>
      <div class='metric-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATH-IN-AI TAGS
# ─────────────────────────────────────────────
st.markdown("""
<div class='info-box'>
  <strong>Math in AI — Where it lives in this app:</strong><br><br>
  <span class='math-tag'>MILP · PuLP</span> Objective function maximises weekly protein subject to budget + calorie constraints &nbsp;|&nbsp;
  <span class='math-tag'>Harris-Benedict</span> Biometric equation calculates personalised calorie targets &nbsp;|&nbsp;
  <span class='math-tag'>Inventory Weighting</span> +100 MILP objective bonus for meals using your pantry items &nbsp;|&nbsp;
  <span class='math-tag'>Shannon Entropy H</span> Measures plan diversity to prevent metabolic adaptation &nbsp;|&nbsp;
  <span class='math-tag'>Operations Research</span> Full OR pipeline: model → solve → interpret
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📅  Weekly Meal Plan", "🔬  The Science"])

# ── TAB 1 — WEEKLY MEAL PLAN ─────────────────
with tab1:
    if "plan" not in st.session_state:
        st.session_state.plan = None
        st.session_state.status = None

    if solve_btn:
        with st.spinner("Running MILP solver (PuLP / CBC)…"):
            plan, status = solve_meal_plan(df, budget, daily_kcal, inventory_list)
        st.session_state.plan   = plan
        st.session_state.status = status

    plan   = st.session_state.plan
    status = st.session_state.status

    if plan is None and not solve_btn:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0;'>
          <div style='font-size:3rem;'>🧮</div>
          <div style='font-family:DM Serif Display,serif;font-size:1.6rem;color:#f5c518;margin:.5rem 0;'>
            Set your parameters and hit Solve
          </div>
          <div style='color:#7a8099;font-size:.9rem;'>
            The MILP engine will generate your optimal 21-meal weekly plan.
          </div>
        </div>
        """, unsafe_allow_html=True)
    elif status and status != "Optimal":
        st.error(f"Solver returned: **{status}**. Try increasing your budget or adjusting parameters.")
    else:
        if plan:
            # ── Summary stats
            total_cost    = sum(df.loc[i, "Cost"]     for i in plan)
            total_protein = sum(df.loc[i, "Protein"]  for i in plan)
            total_kcal    = sum(df.loc[i, "Calories"] for i in plan)
            entropy_val   = shannon_entropy(plan)

            s1, s2, s3, s4 = st.columns(4)
            for col, label, val, good in [
                (s1, "Total Cost",    f"AED {total_cost:.1f}",    total_cost <= budget),
                (s2, "Total Protein", f"{total_protein} g",       True),
                (s3, "Total Calories",f"{total_kcal:,} kcal",     total_kcal >= 0.9*weekly_kcal),
                (s4, "Plan Entropy",  f"H = {entropy_val:.2f}",   entropy_val > 3.0),
            ]:
                colour = "#3af0b0" if good else "#ff5e5b"
                col.markdown(f"""
                <div class='metric-card' style='border-color:{colour}33;'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value' style='font-size:1.5rem;color:{colour};'>{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 7 days × 3 meals
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            for d, day in enumerate(days):
                st.markdown(f"### {day}")
                cols3 = st.columns(3)
                meal_names = ["🌅 Breakfast", "☀️ Lunch", "🌙 Dinner"]

                for m in range(3):
                    idx   = plan[d * 3 + m]
                    row   = df.loc[idx]
                    # inventory highlight
                    inv_match = any(
                        item.lower() in row["Ingredients"].lower()
                        for item in inventory_list
                    )
                    border = "#f5c518" if inv_match else "#252a38"
                    badge  = "🏷️ <b style='color:#f5c518;font-size:.7rem;'>USES YOUR PANTRY</b><br>" if inv_match else ""

                    with cols3[m]:
                        st.markdown(f"""
                        <div class='meal-card' style='border-color:{border};'>
                          <img class='meal-img' src='{row["Image"]}' alt='{row["Name"]}'
                               onerror="this.src='https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400&q=70'"/>
                          <div class='meal-body'>
                            <div style='font-family:Space Mono;font-size:.65rem;color:#7a8099;'>{meal_names[m]}</div>
                            <div class='meal-name'>{row["Name"]}</div>
                            {badge}
                            <div class='meal-macro'>
                              <span class='pill pill-gold'>{row["Calories"]} kcal</span>
                              <span class='pill pill-green'>{row["Protein"]}g protein</span>
                              <span class='pill pill-red'>AED {row["Cost"]:.1f}</span>
                            </div>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                        with st.expander("📖 View Recipe"):
                            st.markdown(f"**Ingredients:** {row['Ingredients']}")
                            st.markdown("---")
                            st.markdown(row["Recipe"])

                st.divider()

# ── TAB 2 — THE SCIENCE ──────────────────────
with tab2:
    st.markdown("## The Science Behind The Math Stack")
    st.markdown(
        "This app was built for the **Mathletics Inter-school Competition 2026** "
        "at Ambassador School, Dubai — themed **'Math in AI'**. Every computational "
        "decision in this planner is grounded in formal mathematics."
    )

    st.divider()
    st.markdown("### ① Mixed-Integer Linear Programming (MILP)")
    st.markdown("""
<div class='info-box'>
<span class='math-tag'>Math in AI — Core Engine</span><br><br>
The heart of The Math Stack is a <b>Mixed-Integer Linear Program</b> solved by the open-source
CBC solver via the <code>PuLP</code> library. This is the same class of mathematics used by
airlines to schedule crews, logistics companies to route deliveries, and hospitals to allocate
resources.

<br><br><b>Formal Model:</b>
<br><br>
<b>Decision Variables:</b> x<sub>i</sub> ∈ {0, 1, 2, 3} — how many times meal i appears in the plan<br><br>
<b>Objective (Maximise):</b>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;Σ (Protein<sub>i</sub> + InventoryBonus<sub>i</sub>) · x<sub>i</sub>
<br><br>
<b>Subject to:</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;[Budget]&nbsp;&nbsp;&nbsp; Σ Cost<sub>i</sub> · x<sub>i</sub> ≤ WeeklyBudget<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Calorie Floor]&nbsp; Σ Calories<sub>i</sub> · x<sub>i</sub> ≥ 0.9 × TargetWeeklyKcal<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Meal Count]&nbsp;&nbsp; Σ x<sub>i</sub> = 21<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Integrity]&nbsp;&nbsp;&nbsp; x<sub>i</sub> ∈ ℤ<sup>+</sup>, x<sub>i</sub> ≤ 3<br>
</div>
""", unsafe_allow_html=True)

    st.markdown("### ② Inventory Weighting — Eliminating Financial Leakage")
    st.markdown("""
<div class='info-box'>
<span class='math-tag'>Math in AI — Objective Coefficient Engineering</span><br><br>
When you list pantry items you already own, the MILP objective coefficient for any meal
containing those items is increased by <b>+100</b>. Because protein values are typically
10–46g per meal, a +100 bonus <i>dominates</i> the objective and forces the solver to
select those meals wherever feasible under the budget constraint.
<br><br>
This eliminates <b>financial leakage</b>: the waste caused by buying ingredients that
duplicate what you already have. Operations Research quantifies this — every AED spent
on duplicated inventory is a constraint violation in disguise.
</div>
""", unsafe_allow_html=True)

    st.markdown("### ③ Harris-Benedict Revised Equation (BMR + TDEE)")
    st.markdown(r"""
<div class='info-box'>
<span class='math-tag'>Math in AI — Personalised Calorie Science</span><br><br>
Calorie targets are not arbitrary. They are computed from your biometric inputs using the
<b>Harris-Benedict Revised equations</b> (Roza & Shizgal, 1984):
<br><br>
<b>Male BMR</b> = 88.362 + 13.397·W + 4.799·H − 5.677·A<br>
<b>Female BMR</b> = 447.593 + 9.247·W + 3.098·H − 4.330·A<br><br>
(W = weight kg, H = height cm, A = age years)<br><br>
Then multiplied by a <b>Physical Activity Level (PAL)</b> factor (1.2 – 1.9) to give
Total Daily Energy Expenditure (TDEE). The MILP calorie floor is set at 90% of
(TDEE × 7) to allow solver flexibility while preventing metabolic deficit.
</div>
""", unsafe_allow_html=True)

    st.markdown("### ④ Shannon Entropy — Preventing Metabolic Slowdown")
    st.markdown(r"""
<div class='info-box'>
<span class='math-tag'>Math in AI — Information Theory Applied to Nutrition</span><br><br>
A plan that repeats the same 3 meals every day is nutritionally dangerous — it leads to
micronutrient gaps and metabolic adaptation (the body down-regulates TDEE when it
"learns" the pattern). We measure plan diversity using <b>Shannon Entropy</b>:
<br><br>
<b>H = − Σ p<sub>i</sub> · log<sub>2</sub>(p<sub>i</sub>)</b>
<br><br>
where p<sub>i</sub> = (times meal i appears) / 21. Maximum entropy for 21 meals is
log<sub>2</sub>(21) ≈ 4.39 bits. A healthy plan scores H > 3.0 bits. The integer
cap (x<sub>i</sub> ≤ 3) in the MILP is designed to keep entropy high — no meal
can dominate more than 14.3% of the plan.
</div>
""", unsafe_allow_html=True)

    st.markdown("### ⑤ Operations Research — The Full Pipeline")
    st.markdown("""
<div class='info-box'>
<span class='math-tag'>Math in AI — OR Methodology</span><br><br>
Operations Research (OR) is the mathematical discipline of optimal decision-making.
The Math Stack follows the classic OR pipeline:<br><br>
<b>1. Problem Formulation</b> → Define decision variables (which meals to include),
objective (maximise protein), and constraints (budget, calories, count).<br><br>
<b>2. Mathematical Modelling</b> → Encode as a MILP with 200 binary-integer variables.<br><br>
<b>3. Algorithm Selection</b> → CBC (Coin-or Branch and Cut) uses Branch-and-Bound with
LP relaxations to efficiently search the 3<sup>200</sup> feasible space.<br><br>
<b>4. Solution Interpretation</b> → Extract x<sub>i</sub> values, compute summary
statistics (cost, protein, entropy), flag inventory matches.<br><br>
<b>5. Sensitivity</b> → Adjusting the budget slider is a <i>right-hand-side sensitivity</i>
analysis — the solver re-runs and the optimal basis changes.<br><br>
This pipeline is identical to how airline revenue management systems price seats
or how supply chains decide what to stock. The math is the same; only the data differs.
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='text-align:center;font-family:Space Mono;font-size:.7rem;color:#7a8099;padding:1.5rem 0;'>
    THE MATH STACK &nbsp;·&nbsp; AMBASSADOR SCHOOL MATHLETICS 2026 &nbsp;·&nbsp; MATH IN AI<br>
    Built with PuLP · Streamlit · Pandas · NumPy · SciPy
    </div>
    """, unsafe_allow_html=True)
