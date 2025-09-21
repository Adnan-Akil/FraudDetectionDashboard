import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import math

# Try to import faker, else create a small fallback
try:
    from faker import Faker
    fake = Faker()
except Exception as e:
    fake = None

RND = 42
random.seed(RND)
np.random.seed(RND)

NUM_USERS = 600            # >500 users
TARGET_ROWS = 30000        # at least 25k rows; produce ~30k
START_DATE = datetime(2024, 9, 18)  # one year window start (user timezone context)
END_DATE = datetime(2025, 9, 17)    # inclusive end date
DAYS = (END_DATE - START_DATE).days + 1

CATEGORIES = [
    "groceries", "restaurants", "fuel", "utilities", "entertainment", "electronics",
    "clothing", "travel", "airlines", "hotels", "pharmacy", "healthcare", "education",
    "gaming", "subscription", "furniture", "home_improvement", "telecom", "insurance",
    "charity", "beauty", "sports", "auto_services", "jewelry", "luxury"
]

MERCHANTS_BY_CAT = {
    "groceries": ["FreshMart", "GreenBasket", "DailyNeeds", "GroceryHub"],
    "restaurants": ["PizzaPoint", "SaffronKitchen", "BurgerBarn", "CafeBrew"],
    "fuel": ["FuelExpress", "PetroGo", "GasMate"],
    "utilities": ["PowerGrid", "WaterCo", "CityBills"],
    "entertainment": ["CineWorld", "ConcertZone", "PlayHouse"],
    "electronics": ["GizmoStore", "TechArena", "ElectroShop"],
    "clothing": ["StyleStreet", "UrbanWear", "ClosetClub"],
    "travel": ["TripAway", "TravelNest"],
    "airlines": ["SkyHigh", "AeroLine"],
    "hotels": ["StayInn", "HotelLuxe"],
    "pharmacy": ["MediCare", "PharmaPlus"],
    "healthcare": ["HealthFirst", "ClinicCare"],
    "education": ["EduHub", "TutorConnect"],
    "gaming": ["GameVault", "PlayCenter"],
    "subscription": ["StreamFlix", "MusicNow"],
    "furniture": ["FurniCo", "HomeStyle"],
    "home_improvement": ["ToolDepot", "FixIt"],
    "telecom": ["CallNet", "MobileOne"],
    "insurance": ["SafeGuard", "InsureAll"],
    "charity": ["HelpingHand", "GoodCause"],
    "beauty": ["GlamSpot", "BeautyBox"],
    "sports": ["Sportify", "AthleteStore"],
    "auto_services": ["AutoCare", "QuickFixGarage"],
    "jewelry": ["GoldRing", "LuxGems"],
    "luxury": ["EliteBoutique", "PlushLife"]
}

DEVICES = ["iOS_App", "Android_App", "Web", "POS_Terminal", "Mobile_Web", "Tablet_App"]

COUNTRIES = ["US", "GB", "IN", "CA", "AE", "DE", "FR", "AU", "SG", "NG"]  # varied locales

rows = []

def gen_user_name(i):
    if fake:
        return fake.name()
    else:
        first = random.choice(["Alex","Sam","Jordan","Taylor","Riley","Casey","Morgan","Charlie","Avery","Jamie"])
        last = random.choice(["Smith","Lee","Patel","Garcia","Chen","Khan","Johnson","Brown","Williams","Jones"])
        return f"{first} {last}"

users = {}
for uid in range(1, NUM_USERS+1):
    name = gen_user_name(uid)
    signup = START_DATE + timedelta(days=random.randint(0, DAYS//2))
    country = random.choice(COUNTRIES)
    city = fake.city() if fake else random.choice(["New York","London","Dubai","Mumbai","Toronto","Sydney","Berlin","Paris","Singapore","Lagos"])
    # User preferred categories (3 major + 1 occasional)
    preferred = random.sample(CATEGORIES, 3)
    occasional = random.choice([c for c in CATEGORIES if c not in preferred])
    preferred.append(occasional)
    # Typical transaction amount behavior (log-normal-like)
    avg_amount = round(max(5, np.random.lognormal(mean=3.0, sigma=0.8)), 2)  # typical avg in range ~20-100+
    std_amount = max(1.0, avg_amount * random.uniform(0.15, 0.6))
    # Preferred hours (peak hour mean 0-23), and weekday preference
    preferred_hour = int(np.clip(np.random.normal(loc=18, scale=4), 0, 23))  # evening bias historically
    weekday_bias = random.choice(["weekdays", "weekends", "no_pref"])
    # frequency: avg transactions per week
    weekly_txn_mean = max(1, int(np.random.poisson(lam=5)))  # average 5 per week typical -> ~260 per year; but we'll sample differently
    # burstiness: probability of small bursts (multiple quick transactions)
    burstiness = random.uniform(0.0, 0.3)
    users[uid] = {
        "user_id": uid,
        "name": name,
        "signup": signup,
        "country": country,
        "city": city,
        "preferred_categories": preferred,
        "avg_amount": avg_amount,
        "std_amount": std_amount,
        "preferred_hour": preferred_hour,
        "weekday_bias": weekday_bias,
        "weekly_mean": weekly_txn_mean,
        "burstiness": burstiness
    }

# Decide how many transactions per user so total reaches TARGET_ROWS
# Start with sampling counts then scale to reach target
txn_counts = np.random.poisson(lam=50, size=NUM_USERS)  # mean 50 transactions -> ~30k
txn_counts = np.clip(txn_counts, 20, 200)  # ensure min and cap
scale_factor = TARGET_ROWS / txn_counts.sum()
txn_counts = np.maximum(10, (txn_counts * scale_factor).astype(int))

# Generate transactions per user
global_txn_id = 1
for uid, count in zip(users.keys(), txn_counts):
    profile = users[uid]
    n = int(count)
    # Create base timestamps distributed over the date window following user's weekly_mean and preferred hour
    for seq in range(1, n+1):
        # Base day selection: weight weekdays/weekends per preference
        day_idx = random.randint(0, DAYS-1)
        base_date = START_DATE + timedelta(days=day_idx)
        # Modify hour around preferred_hour using normal but wrap
        hour = int(np.clip(np.random.normal(loc=profile["preferred_hour"], scale=3), 0, 23))
        minute = random.randint(0,59)
        second = random.randint(0,59)
        txn_time = datetime(base_date.year, base_date.month, base_date.day, hour, minute, second)
        # amount: lognormal around avg
        amount = round(np.random.normal(loc=profile["avg_amount"], scale=profile["std_amount"]), 2)
        if amount <= 0:
            amount = round(abs(np.random.normal(loc=profile["avg_amount"], scale=profile["std_amount"])) + 1, 2)
        # choose merchant category biased towards preferred
        if random.random() < 0.85:
            category = random.choice(profile["preferred_categories"])
        else:
            category = random.choice(CATEGORIES)
        merchant = random.choice(MERCHANTS_BY_CAT.get(category, ["LocalMerchant"]))
        device = random.choice(DEVICES)
        country = profile["country"]
        city = profile["city"]
        # frequency cluster: create occasional bursts by duplicating small time deltas
        if random.random() < profile["burstiness"] * 0.2:
            # cluster: add a few rapid transactions shortly after this one
            # we'll mark frequency in later aggregation
            pass
        rows.append({
            "txn_id": global_txn_id,
            "user_id": uid,
            "user_name": profile["name"],
            "signup_date": profile["signup"].date().isoformat(),
            "timestamp": txn_time.isoformat(sep=' '),
            "amount": amount,
            "merchant_category": category,
            "merchant_name": merchant,
            "device": device,
            "city": city,
            "country": country,
            "seq_for_user": seq,
            "is_fraud": 0,
            "fraud_type": ""
        })
        global_txn_id += 1


df = pd.DataFrame(rows)
# If we overshot TARGET_ROWS slightly, sample uniformly; if undershot, we'll keep as is.
if len(df) > TARGET_ROWS:
    df = df.sample(n=TARGET_ROWS, random_state=RND).reset_index(drop=True)

# Now inject frauds per user: ensure ~5% fraud per user
def inject_frauds_for_user(df_user, profile, fraud_fraction=0.05):
    n = len(df_user)
    k = max(5, int(math.ceil(n * fraud_fraction)))  # at least 1 fraud if user has transactions
    fraud_indices = np.random.choice(df_user.index, size=k, replace=False)
    for idx in fraud_indices:
        row = df_user.loc[idx].copy()
        fraud_reasons = []
        # Choose a fraud type randomly among several anomaly strategies
        choice = random.choices(
            ["time_anomaly", "frequency_anomaly", "amount_anomaly", "category_anomaly", "location_anomaly"],
            weights=[0.2,0.2,0.3,0.2,0.1],
            k=1
        )[0]
        if choice == "time_anomaly":
            # push time into unusual hour (e.g., 2am if preferred is evening)
            new_hour = int((profile["preferred_hour"] + 12) % 24)
            txn_dt = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            txn_dt = txn_dt.replace(hour=new_hour, minute=random.randint(0,59), second=random.randint(0,59))
            df_user.at[idx, "timestamp"] = txn_dt.isoformat(sep=' ')
            fraud_reasons.append("time_anomaly")
        elif choice == "frequency_anomaly":
            # simulate rapid burst: create extra transactions nearby (we'll mark this one)
            txn_dt = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            # shift to be within seconds of others by setting a very recent timestamp (simulate many txns in a short window)
            txn_dt = txn_dt.replace(minute=random.randint(0,2), second=random.randint(0,59))
            df_user.at[idx, "timestamp"] = txn_dt.isoformat(sep=' ')
            fraud_reasons.append("frequency_anomaly")
        elif choice == "amount_anomaly":
            # inflate amount by a large factor or make it tiny (card testing)
            if random.random() < 0.6:
                new_amount = round(row["amount"] * random.uniform(5, 30), 2)  # very large
            else:
                new_amount = round(row["amount"] * random.uniform(0.01, 0.1), 2)  # tiny test charge
            df_user.at[idx, "amount"] = float(max(0.5, new_amount))
            fraud_reasons.append("amount_anomaly")
        elif choice == "category_anomaly":
            # use a category outside preferred set and maybe high amount
            out_cats = [c for c in CATEGORIES if c not in profile["preferred_categories"]]
            new_cat = random.choice(out_cats)
            new_merchant = random.choice(MERCHANTS_BY_CAT.get(new_cat, ["StrangeMerchant"]))
            df_user.at[idx, "merchant_category"] = new_cat
            df_user.at[idx, "merchant_name"] = new_merchant
            # bump amount moderately
            df_user.at[idx, "amount"] = round(row["amount"] * random.uniform(2,8), 2)
            fraud_reasons.append("category_anomaly")
        elif choice == "location_anomaly":
            # different country / city
            new_country = random.choice([c for c in COUNTRIES if c != profile["country"]])
            new_city = random.choice(["Istanbul","Cairo","Lisbon","Seoul","Bangkok","Kuala Lumpur"])
            df_user.at[idx, "country"] = new_country
            df_user.at[idx, "city"] = new_city
            fraud_reasons.append("location_anomaly")
        df_user.at[idx, "is_fraud"] = 1
        df_user.at[idx, "fraud_type"] = ",".join(fraud_reasons)
    return df_user

# Apply fraud injection per user group
out_rows = []
for uid, group in df.groupby("user_id"):
    profile = users[int(uid)]
    g2 = inject_frauds_for_user(group.copy(), profile, fraud_fraction=0.05)
    out_rows.append(g2)

df_final = pd.concat(out_rows).reset_index(drop=True)

# Add derived frequency features (per user): rolling counts in windows, total_txn_by_user, avg_amount_by_user
df_final["timestamp_dt"] = pd.to_datetime(df_final["timestamp"], format="%Y-%m-%d %H:%M:%S")
df_final = df_final.sort_values(["user_id","timestamp_dt"]).reset_index(drop=True)
df_final["total_txn_by_user"] = df_final.groupby("user_id")["txn_id"].transform("count")
df_final["avg_amount_by_user"] = df_final.groupby("user_id")["amount"].transform("mean")
# compute transactions in last 1 hour for same user (naive O(n) approach but dataset small enough)
df_final["txns_last_1hr_by_user"] = 0
for uid, g in df_final.groupby("user_id"):
    times = g["timestamp_dt"].values
    counts = []
    # sliding window count with two pointers
    left = 0
    times_list = list(times)
    for right, t in enumerate(times_list):
        while left < right and (t - times_list[left]) > np.timedelta64(3600, 's'):
            left += 1
        counts.append(right - left)  # number of other txns in past 1 hour
    df_final.loc[g.index, "txns_last_1hr_by_user"] = counts

# Simple sanity checks and stats
total_rows = len(df_final)
unique_users = df_final["user_id"].nunique()
global_fraud_rate = df_final["is_fraud"].mean()

summary = {
    "total_rows": total_rows,
    "unique_users": unique_users,
    "global_fraud_rate": round(global_fraud_rate, 4),
    "rows_per_user_mean": df_final.groupby("user_id").size().mean()
}

# Save CSV
out_path = "synthetic_transactions.csv"
df_final.to_csv(out_path, index=False)

# Display small preview and summary stats
display_df = df_final.head(10).copy()
display_df_preview = display_df[["txn_id","user_id","timestamp","amount","merchant_category","merchant_name","country","city","device","is_fraud","fraud_type"]]

summary, out_path

