import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# A list of common merchant categories and a few "high-risk" ones
MERCHANT_CATEGORIES = {
    'grocery': {'names': ['Whole Foods', 'Trader Joe\'s', 'Kroger'], 'amount_range': (20.0, 150.0)},
    'gas': {'names': ['Shell', 'Exxon', 'Chevron'], 'amount_range': (30.0, 70.0)},
    'retail': {'names': ['Amazon', 'Target', 'Walmart'], 'amount_range': (10.0, 300.0)},
    'entertainment': {'names': ['Netflix', 'Spotify', 'AMC Theatres'], 'amount_range': (5.0, 50.0)},
    'travel': {'names': ['Delta Airlines', 'Uber', 'Booking.com'], 'amount_range': (100.0, 1500.0)},
    'high_risk': {'names': ['Online Gambling Site', 'International Money Transfer', 'High-End Electronics'], 'amount_range': (500.0, 5000.0)}
}

class User:
    def __init__(self, fake):
        self.user_id = fake.uuid4()
        self.card_number = fake.credit_card_number()
        self.name = fake.name()
        self.location = fake.city()
        self.normal_start_hour = random.randint(7, 12)
        self.normal_end_hour = random.randint(20, 23)

        self.spending_profile = random.choices(
            list(MERCHANT_CATEGORIES.keys()),
            weights=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05], # Weights for each category
            k=random.randint(2, 4) # Each user has 2-4 primary spending categories
        )
        self.transactions_per_day = random.randint(1, 5)

def create_high_velocity_fraud(user, fake):
    """Generates a series of high-velocity fraudulent transactions."""
    num_fraudulent_txns = random.randint(5, 15)  # Between 5 and 15 transactions
    start_time = fake.date_time_between(start_date='-1y', end_date='now')
    
    fraud_transactions = []
    
    for _ in range(num_fraudulent_txns):
        transaction_time = start_time + timedelta(seconds=random.randint(10, 120))
        
        # Consistent merchant for a velocity attack
        merchant_cat = 'retail'
        amount_range = MERCHANT_CATEGORIES[merchant_cat]['amount_range']
        
        transaction = {
            'transaction_id': fake.uuid4(),
            'user_id': user.user_id,
            'card_number': user.card_number,
            'transaction_date': transaction_time,
            'amount': round(random.uniform(amount_range[0], amount_range[1]), 2),
            'merchant_name': random.choice(MERCHANT_CATEGORIES[merchant_cat]['names']),
            'location': fake.city(),
            'merchant_category': merchant_cat,
            'is_fraud': 1,
        }
        fraud_transactions.append(transaction)
        start_time = transaction_time
        
    return fraud_transactions

def generate_synthetic_transactions_with_patterns(num_transactions, num_users, fraud_rate=0.03):
    fake = Faker('en_US')
    users = [User(fake) for _ in range(num_users)]
    all_transactions = []
    
    num_fraud_events = int(num_transactions * fraud_rate / 2)
    
    for _ in range(num_fraud_events):
        user = random.choice(users)
        pattern = random.choice(['velocity', 'timing', 'sequence', 'category'])
        
        if pattern == 'velocity':
            all_transactions.extend(create_high_velocity_fraud(user, fake))
        elif pattern == 'timing':
            transaction_date = fake.date_time_between(start_date='-1y', end_date='now').replace(hour=random.randint(1, 5))
            merchant_cat = random.choice(list(MERCHANT_CATEGORIES.keys()))
            amount_range = MERCHANT_CATEGORIES[merchant_cat]['amount_range']
            
            all_transactions.append({
                'transaction_id': fake.uuid4(), 'user_id': user.user_id, 'card_number': user.card_number, 
                'transaction_date': transaction_date, 'amount': round(random.uniform(amount_range[0], amount_range[1]), 2),
                'merchant_name': random.choice(MERCHANT_CATEGORIES[merchant_cat]['names']), 'location': user.location, 'merchant_category': merchant_cat, 'is_fraud': 1
            })
        elif pattern == 'sequence':
            large_transaction_date = fake.date_time_between(start_date='-1y', end_date='now')
            small_transaction_date = large_transaction_date - timedelta(minutes=random.randint(1, 10))
            
            # Small transaction has a small amount
            small_amount_range = (1.0, 10.0)
            merchant_cat_small = 'retail'
            
            # Large transaction has a high-risk amount
            large_amount_range = MERCHANT_CATEGORIES['high_risk']['amount_range']
            merchant_cat_large = 'high_risk'
            
            all_transactions.extend([
                {'transaction_id': fake.uuid4(), 'user_id': user.user_id, 'card_number': user.card_number, 'transaction_date': small_transaction_date, 'amount': round(random.uniform(small_amount_range[0], small_amount_range[1]), 2), 'merchant_name': random.choice(MERCHANT_CATEGORIES[merchant_cat_small]['names']), 'location': fake.city(), 'merchant_category': merchant_cat_small, 'is_fraud': 1},
                {'transaction_id': fake.uuid4(), 'user_id': user.user_id, 'card_number': user.card_number, 'transaction_date': large_transaction_date, 'amount': round(random.uniform(large_amount_range[0], large_amount_range[1]), 2), 'merchant_name': random.choice(MERCHANT_CATEGORIES[merchant_cat_large]['names']), 'location': fake.city(), 'merchant_category': merchant_cat_large, 'is_fraud': 1}
            ])
        elif pattern == 'category':
            merchant_cat = 'high_risk'
            amount_range = MERCHANT_CATEGORIES[merchant_cat]['amount_range']
            all_transactions.append({
                'transaction_id': fake.uuid4(), 'user_id': user.user_id, 'card_number': user.card_number, 
                'transaction_date': fake.date_time_between(start_date='-1y', end_date='now'), 'amount': round(random.uniform(amount_range[0], amount_range[1]), 2),
                'merchant_name': random.choice(MERCHANT_CATEGORIES[merchant_cat]['names']), 'location': user.location, 'merchant_category': merchant_cat, 'is_fraud': 1
            })

    # Fill the rest of the transactions with non-fraudulent data
    num_normal_transactions = num_transactions - len(all_transactions)
    for _ in range(num_normal_transactions):
        user = random.choice(users)
        
        # Pick a merchant category from the user's spending profile
        merchant_cat = random.choice(user.spending_profile)
        # Get a realistic amount based on the category
        amount_range = MERCHANT_CATEGORIES[merchant_cat]['amount_range']
        
        all_transactions.append({
            'transaction_id': fake.uuid4(), 'user_id': user.user_id, 'card_number': user.card_number,
            'transaction_date': fake.date_time_between(start_date='-1y', end_date='now'), 'amount': round(random.uniform(amount_range[0], amount_range[1]), 2),
            'merchant_name': random.choice(MERCHANT_CATEGORIES[merchant_cat]['names']), 'location': user.location,
            'merchant_category': merchant_cat, 'is_fraud': 0
        })

    df = pd.DataFrame(all_transactions)
    df = df.sort_values(by='transaction_date').reset_index(drop=True)
    return df

if __name__ == '__main__':
    num_records = 10000
    num_unique_users = 1000
    df_transactions = generate_synthetic_transactions_with_patterns(num_records, num_unique_users)
    
    file_path = 'synthetic_transactions_with_correlations.csv'
    df_transactions.to_csv(file_path, index=False)
    
    print(f"Successfully generated {len(df_transactions)} synthetic transactions for {num_unique_users} users.")
    print(f"Data saved to {file_path}")
    print("\nSample of the generated data:")
    print(df_transactions.head())