import pandas as pd
import numpy as np
from datetime import datetime, time
import re
from typing import Dict, List, Tuple
import random

class DataProcessor:
    def __init__(self):
        self.required_columns = [
            'TransactionID', 'UserID', 'Amount', 
            'Time', 'Location', 'Type'
        ]
    
    def validate_csv(self, filepath: str) -> Tuple[bool, str, pd.DataFrame]:
        """Validate CSV file format and content"""
        try:
            df = pd.read_csv(filepath)
            
            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                return False, f"Missing columns: {', '.join(missing_cols)}", None
            
            # Check for empty dataframe
            if df.empty:
                return False, "CSV file is empty", None
            
            # Basic data validation
            if df['Amount'].dtype not in ['int64', 'float64']:
                try:
                    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
                    if df['Amount'].isnull().any():
                        return False, "Amount column contains non-numeric values", None
                except:
                    return False, "Amount column must contain numeric values", None
            
            # Check for required data
            if df['TransactionID'].isnull().any():
                return False, "TransactionID cannot be empty", None
            
            return True, "CSV validation successful", df
            
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}", None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['TransactionID'])
        
        # Clean amount column
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna(subset=['Amount'])
        
        # Standardize time format
        df['Time'] = df['Time'].apply(self._standardize_time)
        
        # Clean location
        df['Location'] = df['Location'].str.strip().str.title()
        
        # Clean transaction type
        df['Type'] = df['Type'].str.strip().str.upper()
        
        return df
    
    def _standardize_time(self, time_str):
        """Standardize time format to HH:MM"""
        try:
            # Handle various time formats
            time_str = str(time_str).strip()
            
            # Try parsing different formats
            formats = ['%H:%M', '%H:%M:%S', '%I:%M %p', '%I:%M:%S %p']
            
            for fmt in formats:
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    return parsed_time.strftime('%H:%M')
                except ValueError:
                    continue
            
            # If no format works, try extracting HH:MM pattern
            match = re.search(r'(\d{1,2}):(\d{2})', time_str)
            if match:
                hour, minute = match.groups()
                return f"{int(hour):02d}:{minute}"
            
            return "12:00"  # Default fallback
            
        except:
            return "12:00"  # Default fallback
    
    def build_user_profiles(self, df: pd.DataFrame) -> Dict:
        """Build user behavior profiles from historical data"""
        profiles = {}
        
        for user_id in df['UserID'].unique():
            user_data = df[df['UserID'] == user_id]
            
            # Calculate statistics
            amounts = user_data['Amount']
            times = pd.to_datetime(user_data['Time'], format='%H:%M').dt.hour
            locations = user_data['Location'].value_counts()
            
            profile = {
                'user_id': user_id,
                'avg_amount': amounts.mean(),
                'std_amount': amounts.std() if len(amounts) > 1 else amounts.mean() * 0.1,
                'median_amount': amounts.median(),
                'min_amount': amounts.min(),
                'max_amount': amounts.max(),
                'normal_hours_start': max(6, times.quantile(0.1)),  # 10th percentile, min 6 AM
                'normal_hours_end': min(22, times.quantile(0.9)),   # 90th percentile, max 10 PM
                'common_locations': locations.head(3).index.tolist(),
                'transaction_frequency': len(user_data),
                'preferred_types': user_data['Type'].value_counts().head(2).index.tolist()
            }
            
            profiles[user_id] = profile
        
        return profiles
    
    def extract_features(self, transaction: pd.Series, user_profiles: Dict) -> List[float]:
        """Extract features for anomaly detection"""
        user_id = transaction['UserID']
        profile = user_profiles.get(user_id, self._get_default_profile())
        
        features = []
        
        # Amount-based features
        if profile['avg_amount'] > 0:
            features.append(transaction['Amount'] / profile['avg_amount'])  # Amount ratio
            if profile['std_amount'] > 0:
                features.append(abs(transaction['Amount'] - profile['avg_amount']) / profile['std_amount'])  # Z-score
            else:
                features.append(0.0)
        else:
            features.extend([1.0, 0.0])
        
        # Time-based features
        transaction_hour = int(transaction['Time'].split(':')[0])
        
        # Hour of day (normalized)
        features.append(transaction_hour / 24.0)
        
        # Is outside normal hours
        is_outside_hours = 1.0 if (transaction_hour < profile['normal_hours_start'] or 
                                  transaction_hour > profile['normal_hours_end']) else 0.0
        features.append(is_outside_hours)
        
        # Location-based features
        is_common_location = 1.0 if transaction['Location'] in profile['common_locations'] else 0.0
        features.append(is_common_location)
        
        # Transaction type features
        is_common_type = 1.0 if transaction['Type'] in profile['preferred_types'] else 0.0
        features.append(is_common_type)
        
        # Amount percentile within user's history
        if profile['max_amount'] > profile['min_amount']:
            amount_percentile = (transaction['Amount'] - profile['min_amount']) / (profile['max_amount'] - profile['min_amount'])
            features.append(min(1.0, amount_percentile))
        else:
            features.append(0.5)
        
        return features
    
    def _get_default_profile(self) -> Dict:
        """Get default profile for new users"""
        return {
            'avg_amount': 5000.0,
            'std_amount': 2000.0,
            'median_amount': 3000.0,
            'min_amount': 100.0,
            'max_amount': 10000.0,
            'normal_hours_start': 9,
            'normal_hours_end': 18,
            'common_locations': ['Kathmandu'],
            'transaction_frequency': 10,
            'preferred_types': ['TRANSFER', 'WITHDRAWAL']
        }
    
    def generate_sample_data(self, num_transactions: int = 100) -> pd.DataFrame:
        """Generate sample transaction data for demo"""
        try:
            # Set seed for reproducible results
            random.seed(42)
            
            # Sample users
            users = [f"U{i:03d}" for i in range(1, 21)]  # 20 users
            locations = ['Kathmandu', 'Pokhara', 'Lalitpur', 'Bhaktapur', 'Chitwan']
            transaction_types = ['TRANSFER', 'WITHDRAWAL', 'DEPOSIT', 'PAYMENT']
            
            transactions = []
            
            for i in range(num_transactions):
                user_id = random.choice(users)
                
                # Generate normal transaction (90%)
                if i < int(num_transactions * 0.9):
                    amount = random.uniform(1000, 15000)
                    # Business hours with some variation
                    hour = random.choice([9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
                    minute = random.randint(0, 59)
                    location = random.choice(locations[:3])  # Mostly local locations
                    
                else:  # Generate fraudulent patterns (10%)
                    fraud_type = i % 3
                    
                    if fraud_type == 0:  # Unusual time + high amount
                        amount = random.uniform(50000, 100000)
                        hour = random.choice([2, 3, 4, 23])  # Unusual hours
                        minute = random.randint(0, 59)
                        location = random.choice(locations)
                        
                    elif fraud_type == 1:  # Test pattern (small amounts)
                        amount = random.uniform(1, 10)
                        hour = random.choice([3, 4])
                        minute = random.randint(45, 59)
                        location = random.choice(locations)
                        
                    else:  # Unusual location + medium amount
                        amount = random.uniform(15000, 30000)
                        hour = random.choice(range(9, 19))
                        minute = random.randint(0, 59)
                        location = 'Delhi'  # Unusual location
                
                transaction = {
                    'TransactionID': f"T{i+1:04d}",
                    'UserID': user_id,
                    'Amount': round(amount, 2),
                    'Time': f"{hour:02d}:{minute:02d}",
                    'Location': location,
                    'Type': random.choice(transaction_types)
                }
                
                transactions.append(transaction)
            
            return pd.DataFrame(transactions)
            
        except Exception as e:
            # Fallback to simple hardcoded data
            print(f"Error in generate_sample_data: {e}")
            return self._create_fallback_data()
    
    def _create_fallback_data(self) -> pd.DataFrame:
        """Create simple fallback data if generation fails"""
        fallback_transactions = [
            {'TransactionID': 'T001', 'UserID': 'U001', 'Amount': 2500.00, 'Time': '14:30', 'Location': 'Kathmandu', 'Type': 'TRANSFER'},
            {'TransactionID': 'T002', 'UserID': 'U002', 'Amount': 1800.50, 'Time': '10:15', 'Location': 'Pokhara', 'Type': 'WITHDRAWAL'},
            {'TransactionID': 'T003', 'UserID': 'U003', 'Amount': 75000.00, 'Time': '03:47', 'Location': 'Kathmandu', 'Type': 'TRANSFER'},  # FRAUD
            {'TransactionID': 'T004', 'UserID': 'U001', 'Amount': 1.00, 'Time': '03:48', 'Location': 'Kathmandu', 'Type': 'TRANSFER'},      # FRAUD
            {'TransactionID': 'T005', 'UserID': 'U001', 'Amount': 1.00, 'Time': '03:49', 'Location': 'Kathmandu', 'Type': 'TRANSFER'},      # FRAUD
            {'TransactionID': 'T006', 'UserID': 'U001', 'Amount': 50000.00, 'Time': '03:50', 'Location': 'Kathmandu', 'Type': 'TRANSFER'}, # FRAUD
            {'TransactionID': 'T007', 'UserID': 'U004', 'Amount': 3200.75, 'Time': '15:20', 'Location': 'Lalitpur', 'Type': 'DEPOSIT'},
            {'TransactionID': 'T008', 'UserID': 'U005', 'Amount': 4500.00, 'Time': '12:30', 'Location': 'Kathmandu', 'Type': 'PAYMENT'},
            {'TransactionID': 'T009', 'UserID': 'U002', 'Amount': 890.25, 'Time': '16:45', 'Location': 'Pokhara', 'Type': 'WITHDRAWAL'},
            {'TransactionID': 'T010', 'UserID': 'U006', 'Amount': 95000.00, 'Time': '02:15', 'Location': 'Delhi', 'Type': 'TRANSFER'},       # FRAUD
            {'TransactionID': 'T011', 'UserID': 'U007', 'Amount': 2100.00, 'Time': '11:30', 'Location': 'Kathmandu', 'Type': 'TRANSFER'},
            {'TransactionID': 'T012', 'UserID': 'U003', 'Amount': 3800.50, 'Time': '14:00', 'Location': 'Pokhara', 'Type': 'DEPOSIT'},
            {'TransactionID': 'T013', 'UserID': 'U008', 'Amount': 150000.00, 'Time': '04:22', 'Location': 'Mumbai', 'Type': 'TRANSFER'},    # FRAUD
            {'TransactionID': 'T014', 'UserID': 'U001', 'Amount': 2200.00, 'Time': '13:15', 'Location': 'Kathmandu', 'Type': 'WITHDRAWAL'},
            {'TransactionID': 'T015', 'UserID': 'U009', 'Amount': 1200.75, 'Time': '09:30', 'Location': 'Lalitpur', 'Type': 'PAYMENT'},
            {'TransactionID': 'T016', 'UserID': 'U010', 'Amount': 5.00, 'Time': '23:45', 'Location': 'Kathmandu', 'Type': 'TRANSFER'},       # FRAUD
            {'TransactionID': 'T017', 'UserID': 'U004', 'Amount': 6700.00, 'Time': '16:20', 'Location': 'Pokhara', 'Type': 'DEPOSIT'},
            {'TransactionID': 'T018', 'UserID': 'U011', 'Amount': 85000.00, 'Time': '01:30', 'Location': 'Bangkok', 'Type': 'TRANSFER'},    # FRAUD
            {'TransactionID': 'T019', 'UserID': 'U005', 'Amount': 1950.25, 'Time': '10:45', 'Location': 'Kathmandu', 'Type': 'WITHDRAWAL'},
            {'TransactionID': 'T020', 'UserID': 'U012', 'Amount': 3400.00, 'Time': '15:30', 'Location': 'Lalitpur', 'Type': 'PAYMENT'}
        ]
        
        return pd.DataFrame(fallback_transactions)