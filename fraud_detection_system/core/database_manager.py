import sqlite3
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from config.settings import DATABASE_PATH

class DatabaseManager:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE,
                    user_id TEXT,
                    amount REAL,
                    timestamp TEXT,
                    location TEXT,
                    transaction_type TEXT,
                    ai_prediction INTEGER,
                    ai_confidence REAL,
                    user_feedback INTEGER,
                    feedback_timestamp TEXT,
                    features_json TEXT,
                    model_version TEXT,
                    session_id TEXT
                )
            ''')
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    avg_amount REAL,
                    std_amount REAL,
                    normal_hours_start INTEGER,
                    normal_hours_end INTEGER,
                    common_locations TEXT,
                    transaction_frequency REAL,
                    last_updated TEXT
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    training_date TEXT,
                    samples_count INTEGER,
                    feedback_count INTEGER
                )
            ''')
            
            # Analysis sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT,
                    total_transactions INTEGER,
                    flagged_count INTEGER,
                    feedback_provided INTEGER,
                    analysis_date TEXT,
                    model_version TEXT
                )
            ''')
            
            conn.commit()
    
    def save_transaction_analysis(self, session_id, results):
        """Save transaction analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for result in results:
                cursor.execute('''
                    INSERT OR REPLACE INTO transactions 
                    (transaction_id, user_id, amount, timestamp, location, 
                     transaction_type, ai_prediction, ai_confidence, 
                     features_json, model_version, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['transaction_id'],
                    result['user_id'],
                    result['amount'],
                    result['timestamp'],
                    result['location'],
                    result['transaction_type'],
                    result['ai_prediction'],
                    result['ai_confidence'],
                    json.dumps(result['features']),
                    result['model_version'],
                    session_id
                ))
            
            conn.commit()
    
    def save_feedback(self, transaction_id, feedback, feedback_type='quick'):
        """Save user feedback for a transaction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE transactions 
                SET user_feedback = ?, feedback_timestamp = ?
                WHERE transaction_id = ?
            ''', (
                1 if feedback == 'correct' else 0,
                datetime.now().isoformat(),
                transaction_id
            ))
            
            conn.commit()
            return cursor.rowcount > 0
    
    def get_feedback_stats(self, session_id=None):
        """Get feedback statistics"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT 
                    COUNT(*) as total_reviewed,
                    SUM(CASE WHEN user_feedback = 1 THEN 1 ELSE 0 END) as confirmed_fraud,
                    SUM(CASE WHEN user_feedback = 0 THEN 1 ELSE 0 END) as confirmed_legitimate,
                    COUNT(*) - COUNT(user_feedback) as pending_review
                FROM transactions 
                WHERE ai_prediction = -1
            '''
            
            if session_id:
                query += f" AND session_id = '{session_id}'"
            
            result = pd.read_sql_query(query, conn)
            return result.iloc[0].to_dict()
    
    def get_training_data(self):
        """Get all feedback data for model training"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT features_json, user_feedback
                FROM transactions 
                WHERE user_feedback IS NOT NULL
            '''
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return None, None
            
            # Parse features
            features = []
            labels = []
            
            for _, row in df.iterrows():
                features.append(json.loads(row['features_json']))
                labels.append(row['user_feedback'])
            
            return features, labels
    
    def save_session(self, session_id, filename, total_count, flagged_count):
        """Save analysis session info"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO analysis_sessions
                (session_id, filename, total_transactions, flagged_count, 
                 feedback_provided, analysis_date, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                filename,
                total_count,
                flagged_count,
                0,  # Will update as feedback comes in
                datetime.now().isoformat(),
                "v1.0"  # Default model version
            ))
            
            conn.commit()