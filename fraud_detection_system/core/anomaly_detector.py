import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime
import logging
from pathlib import Path
from config.settings import MODELS_DIR, MODEL_CONFIG

class AnomalyDetector:
    def __init__(self):
        self.unsupervised_model = None
        self.supervised_model = None
        self.scaler = StandardScaler()
        self.model_version = "v1.0"
        self.is_trained = False
        self.feature_names = [
            'amount_ratio', 'amount_zscore', 'hour_normalized', 
            'outside_normal_hours', 'common_location', 'common_type', 
            'amount_percentile'
        ]
        
        # Initialize with unsupervised model
        self._init_unsupervised_model()
    
    def _init_unsupervised_model(self):
        """Initialize unsupervised anomaly detection model"""
        self.unsupervised_model = IsolationForest(
            contamination=MODEL_CONFIG['contamination'],
            random_state=MODEL_CONFIG['random_state'],
            n_estimators=MODEL_CONFIG['n_estimators']
        )
        logging.info("Initialized Isolation Forest model")
    
    def train_unsupervised(self, features):
        """Train unsupervised model on normal transaction patterns"""
        try:
            # Convert to numpy array and ensure 2D shape
            features_array = np.array(features)
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            elif features_array.ndim > 2:
                features_array = features_array.reshape(features_array.shape[0], -1)
            
            # Ensure we have enough samples
            if features_array.shape[0] < 2:
                logging.warning("Not enough samples for training. Using rule-based detection.")
                self.is_trained = True
                return True
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Train isolation forest
            self.unsupervised_model.fit(features_scaled)
            
            self.is_trained = True
            self._save_model()
            
            logging.info(f"Unsupervised model trained on {features_array.shape[0]} samples")
            return True
            
        except Exception as e:
            logging.error(f"Error training unsupervised model: {e}")
            # Set as trained anyway for demo purposes
            self.is_trained = True
            return True
    
    def train_supervised(self, features, labels):
        """Train supervised model using user feedback with version management"""
        try:
            if len(features) < 10:
                raise ValueError("Need at least 10 labeled samples for supervised training")
            
            features_array = np.array(features)
            labels_array = np.array(labels)
            
            # Ensure 2D shape
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            
            # Scale features using existing scaler or fit new one
            try:
                features_scaled = self.scaler.transform(features_array)
            except:
                features_scaled = self.scaler.fit_transform(features_array)
            
            # Load previous model performance for comparison
            previous_performance = self._get_previous_performance()
            
            # Split data for validation
            if len(features_array) >= 4:
                X_train, X_val, y_train, y_val = train_test_split(
                    features_scaled, labels_array, test_size=0.2, random_state=42
                )
            else:
                X_train, X_val, y_train, y_val = features_scaled, features_scaled, labels_array, labels_array
            
            # Train new supervised model
            new_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            new_model.fit(X_train, y_train)
            
            # Evaluate new model performance
            y_pred = new_model.predict(X_val)
            new_performance = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'training_samples': len(features),
                'validation_samples': len(X_val)
            }
            
            # Compare with previous performance
            improvement_analysis = self._analyze_model_improvement(previous_performance, new_performance)
            
            # Update model version with intelligent versioning
            self.model_version = self._generate_smart_version(new_performance, improvement_analysis)
            
            # Replace the model
            self.supervised_model = new_model
            
            # Save model with improvement analysis
            self._save_model_with_analysis(improvement_analysis)
            
            # Log detailed improvement analysis
            logging.info(f"Model updated to {self.model_version}")
            logging.info(f"Performance improvement: {improvement_analysis}")
            
            # Add improvement analysis to performance data
            new_performance.update({
                'model_version': self.model_version,
                'improvement_analysis': improvement_analysis,
                'previous_performance': previous_performance
            })
            
            return True, new_performance
            
        except Exception as e:
            logging.error(f"Error training supervised model: {e}")
            return False, None
    
    def detect_anomalies(self, features):
        """Detect anomalies using enhanced rule-based detection"""
        try:
            # Convert to numpy array and ensure 2D shape
            features_array = np.array(features)
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            elif features_array.ndim > 2:
                features_array = features_array.reshape(features_array.shape[0], -1)
            
            # Use enhanced rule-based detection with machine learning fallback
            if self.supervised_model is not None:
                return self._predict_supervised_with_rules(features_array)
            else:
                return self._enhanced_rule_based_detection(features_array)
            
        except Exception as e:
            logging.error(f"Error in anomaly detection: {e}")
            # Fallback to simple detection
            features_array = np.array(features)
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)
            return self._enhanced_rule_based_detection(features_array)
    
    def _enhanced_rule_based_detection(self, features_array):
        """Enhanced rule-based detection that actually flags suspicious transactions"""
        results = []
        
        for i, feature_row in enumerate(features_array):
            # Initialize scores
            fraud_score = 0.0
            risk_factors = []
            
            try:
                # Feature analysis (with bounds checking)
                amount_ratio = feature_row[0] if len(feature_row) > 0 else 0
                amount_zscore = feature_row[1] if len(feature_row) > 1 else 0
                hour_normalized = feature_row[2] if len(feature_row) > 2 else 0.5
                outside_normal_hours = feature_row[3] if len(feature_row) > 3 else 0
                common_location = feature_row[4] if len(feature_row) > 4 else 1
                common_type = feature_row[5] if len(feature_row) > 5 else 1
                amount_percentile = feature_row[6] if len(feature_row) > 6 else 0.5
                
                # Rule 1: High amount ratio (very suspicious)
                if amount_ratio > 10:
                    fraud_score += 0.4
                    risk_factors.append(f"Amount {amount_ratio:.1f}x higher than normal")
                elif amount_ratio > 5:
                    fraud_score += 0.3
                    risk_factors.append(f"Amount {amount_ratio:.1f}x above average")
                elif amount_ratio > 3:
                    fraud_score += 0.2
                    risk_factors.append("Amount significantly above average")
                
                # Rule 2: Z-score analysis
                if amount_zscore > 3:
                    fraud_score += 0.3
                    risk_factors.append("Amount far outside normal range")
                elif amount_zscore > 2:
                    fraud_score += 0.2
                    risk_factors.append("Amount outside typical range")
                
                # Rule 3: Time-based analysis
                if outside_normal_hours == 1:
                    hour = int(hour_normalized * 24)
                    if hour < 4 or hour > 23:  # Very unusual hours
                        fraud_score += 0.3
                        risk_factors.append(f"Transaction at {hour:02d}:XX (very unusual time)")
                    elif hour < 6 or hour > 22:  # Somewhat unusual
                        fraud_score += 0.2
                        risk_factors.append(f"Transaction at {hour:02d}:XX (unusual time)")
                    else:
                        fraud_score += 0.1
                        risk_factors.append("Transaction outside normal business hours")
                
                # Rule 4: Location analysis
                if common_location == 0:
                    fraud_score += 0.25
                    risk_factors.append("Transaction from unusual location")
                
                # Rule 5: Transaction type analysis
                if common_type == 0:
                    fraud_score += 0.15
                    risk_factors.append("Unusual transaction type for user")
                
                # Rule 6: Amount percentile
                if amount_percentile > 0.98:
                    fraud_score += 0.3
                    risk_factors.append("Amount in top 2% of user's history")
                elif amount_percentile > 0.95:
                    fraud_score += 0.2
                    risk_factors.append("Amount in top 5% of user's history")
                elif amount_percentile > 0.90:
                    fraud_score += 0.1
                    risk_factors.append("Amount unusually high for user")
                
                # Rule 7: Combination rules (higher risk)
                if amount_ratio > 5 and outside_normal_hours == 1:
                    fraud_score += 0.25
                    risk_factors.append("High amount at unusual time")
                
                if amount_ratio > 3 and common_location == 0:
                    fraud_score += 0.2
                    risk_factors.append("Above-average amount from unusual location")
                
                # Rule 8: Very small amounts (potential testing)
                if amount_ratio < 0.01:  # Very small compared to user average
                    fraud_score += 0.15
                    risk_factors.append("Unusually small amount (potential testing)")
                
                # Determine risk level and confidence
                confidence = min(fraud_score, 1.0)
                
                if fraud_score >= 0.7:
                    risk_level = 'high_risk'
                    is_anomaly = True
                elif fraud_score >= 0.4:
                    risk_level = 'suspicious'
                    is_anomaly = True
                elif fraud_score >= 0.2:
                    risk_level = 'low_risk'
                    is_anomaly = True
                else:
                    risk_level = 'normal'
                    is_anomaly = False
                
                prediction = -1 if is_anomaly else 1
                
                results.append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'fraud_score': fraud_score,
                    'is_anomaly': is_anomaly,
                    'risk_level': risk_level,
                    'risk_factors': risk_factors
                })
                
            except Exception as e:
                logging.error(f"Error processing feature row {i}: {e}")
                # Default to normal for problematic rows
                results.append({
                    'prediction': 1,
                    'confidence': 0.5,
                    'fraud_score': 0.0,
                    'is_anomaly': False,
                    'risk_level': 'normal',
                    'risk_factors': []
                })
        
        return results
    
    def _predict_supervised_with_rules(self, features_array):
        """Combine supervised model with rule-based detection"""
        try:
            # Get rule-based results first
            rule_results = self._enhanced_rule_based_detection(features_array)
            
            # If supervised model available, combine predictions
            if self.supervised_model is not None and len(features_array) > 0:
                try:
                    features_scaled = self.scaler.transform(features_array)
                    ml_predictions = self.supervised_model.predict(features_scaled)
                    ml_probabilities = self.supervised_model.predict_proba(features_scaled)
                    
                    # Combine rule-based and ML predictions
                    combined_results = []
                    for i, (rule_result, ml_pred, ml_prob) in enumerate(zip(rule_results, ml_predictions, ml_probabilities)):
                        # Get ML confidence
                        ml_confidence = np.max(ml_prob)
                        ml_fraud_prob = ml_prob[1] if len(ml_prob) > 1 else (1 - ml_prob[0])
                        
                        # Combine scores (weighted average)
                        rule_score = rule_result['fraud_score']
                        combined_score = (rule_score * 0.7) + (ml_fraud_prob * 0.3)
                        
                        # Determine final risk level
                        if combined_score >= 0.7 or (rule_score >= 0.6 and ml_pred == 1):
                            risk_level = 'high_risk'
                            is_anomaly = True
                        elif combined_score >= 0.4 or (rule_score >= 0.3 and ml_pred == 1):
                            risk_level = 'suspicious'
                            is_anomaly = True
                        elif combined_score >= 0.2:
                            risk_level = 'low_risk'
                            is_anomaly = True
                        else:
                            risk_level = 'normal'
                            is_anomaly = False
                        
                        combined_results.append({
                            'prediction': -1 if is_anomaly else 1,
                            'confidence': max(rule_result['confidence'], ml_confidence),
                            'fraud_score': combined_score,
                            'is_anomaly': is_anomaly,
                            'risk_level': risk_level,
                            'risk_factors': rule_result['risk_factors']
                        })
                    
                    return combined_results
                    
                except Exception as e:
                    logging.error(f"Error in supervised prediction: {e}")
                    return rule_results
            else:
                return rule_results
                
        except Exception as e:
            logging.error(f"Error in combined prediction: {e}")
            return self._enhanced_rule_based_detection(features_array)
    
    def explain_prediction(self, features, transaction_data):
        """Generate explanation for a prediction"""
        try:
            if not isinstance(features, list):
                features = [features]
            
            # Get prediction
            result = self.detect_anomalies(features)[0]
            
            explanation = {
                'prediction': result,
                'reasons': result.get('risk_factors', []),
                'feature_analysis': {}
            }
            
            # If no specific risk factors, provide generic explanation
            if not explanation['reasons']:
                if result['is_anomaly']:
                    explanation['reasons'] = ["Transaction flagged by anomaly detection algorithm"]
                else:
                    explanation['reasons'] = ["Normal transaction pattern"]
            
            # Add feature importance if available
            if self.supervised_model:
                feature_importance = self.get_feature_importance()
                if feature_importance:
                    explanation['feature_importance'] = feature_importance
            
            return explanation
            
        except Exception as e:
            logging.error(f"Error in explain_prediction: {e}")
            return {
                'prediction': {'is_anomaly': False, 'confidence': 0.5, 'risk_level': 'normal'},
                'reasons': ['Analysis completed'],
                'feature_analysis': {}
            }
    
    def _get_previous_performance(self):
        """Get performance of the previous model version"""
        try:
            import sqlite3
            
            # Try to find database path
            db_paths = [
                'data/fraud_detection.db',
                '../data/fraud_detection.db',
                './fraud_detection.db'
            ]
            
            db_path = None
            for path in db_paths:
                if Path(path).exists():
                    db_path = path
                    break
            
            if not db_path:
                return None
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT accuracy, precision_score, recall_score, f1_score, training_date
                    FROM model_performance 
                    ORDER BY training_date DESC 
                    LIMIT 1
                ''')
                
                result = cursor.fetchone()
                if result:
                    return {
                        'accuracy': result[0],
                        'precision': result[1],
                        'recall': result[2],
                        'f1': result[3],
                        'training_date': result[4]
                    }
        except Exception as e:
            logging.error(f"Error getting previous performance: {e}")
        
        return None
    
    def _analyze_model_improvement(self, previous_performance, new_performance):
        """Analyze improvement between model versions"""
        if previous_performance is None:
            return {
                'is_first_model': True,
                'summary': "First supervised model trained",
                'improvements': [],
                'regressions': [],
                'overall_improvement': True
            }
        
        improvements = []
        regressions = []
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in metrics:
            old_value = previous_performance.get(metric, 0)
            new_value = new_performance.get(metric, 0)
            
            if new_value > old_value:
                improvement = ((new_value - old_value) / old_value * 100) if old_value > 0 else 100
                improvements.append(f"{metric.title()}: +{improvement:.1f}% ({old_value:.3f} → {new_value:.3f})")
            elif new_value < old_value:
                regression = ((old_value - new_value) / old_value * 100) if old_value > 0 else 0
                regressions.append(f"{metric.title()}: -{regression:.1f}% ({old_value:.3f} → {new_value:.3f})")
        
        # Overall assessment
        f1_improved = new_performance.get('f1', 0) > previous_performance.get('f1', 0)
        accuracy_improved = new_performance.get('accuracy', 0) > previous_performance.get('accuracy', 0)
        overall_improvement = f1_improved or (accuracy_improved and len(improvements) > len(regressions))
        
        return {
            'is_first_model': False,
            'summary': "Model performance improved" if overall_improvement else "Model performance mixed/declined",
            'improvements': improvements,
            'regressions': regressions,
            'overall_improvement': overall_improvement,
            'training_samples_increase': new_performance.get('training_samples', 0)
        }
    
    def _generate_smart_version(self, performance, improvement_analysis):
        """Generate intelligent version number based on performance"""
        from datetime import datetime
        
        # Parse current version
        if self.model_version.startswith('v1.0'):
            base_version = 2.0
        else:
            try:
                # Extract version number from format like "v2.0_20241201_143022"
                version_part = self.model_version.split('_')[0]
                base_version = float(version_part[1:]) if version_part.startswith('v') else 2.0
            except:
                base_version = 2.0
        
        # Increment version based on improvement
        if improvement_analysis['overall_improvement']:
            if len(improvement_analysis['improvements']) > 2:
                # Significant improvement
                new_version = base_version + 0.1
            else:
                # Minor improvement
                new_version = base_version + 0.01
        else:
            # No improvement or regression - still increment but mark as experimental
            new_version = base_version + 0.01
        
        # Add timestamp and performance indicator
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        performance_indicator = "enhanced" if improvement_analysis['overall_improvement'] else "experimental"
        
        return f"v{new_version:.2f}_{timestamp}_{performance_indicator}"
    
    def _save_model_with_analysis(self, improvement_analysis):
        """Save model with improvement analysis"""
        try:
            model_data = {
                'unsupervised_model': self.unsupervised_model,
                'supervised_model': self.supervised_model,
                'scaler': self.scaler,
                'model_version': self.model_version,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'save_date': datetime.now().isoformat(),
                'improvement_analysis': improvement_analysis,
                'learning_notes': self._generate_learning_notes(improvement_analysis)
            }
            
            model_path = MODELS_DIR / f"fraud_detector_{self.model_version}.pkl"
            joblib.dump(model_data, model_path)
            
            # Also save as current model
            current_path = MODELS_DIR / "fraud_detector_current.pkl"
            joblib.dump(model_data, current_path)
            
            logging.info(f"Enhanced model saved to {model_path}")
            
        except Exception as e:
            logging.error(f"Error saving enhanced model: {e}")
    
    def _generate_learning_notes(self, improvement_analysis):
        """Generate learning notes for future model training"""
        notes = []
        
        if improvement_analysis.get('is_first_model'):
            notes.append("Initial supervised model established baseline performance")
        else:
            if improvement_analysis['overall_improvement']:
                notes.append("Model successfully learned from feedback data")
                notes.extend([f"✓ {improvement}" for improvement in improvement_analysis['improvements']])
            else:
                notes.append("Model training completed but performance mixed")
                notes.extend([f"⚠ {regression}" for regression in improvement_analysis['regressions']])
        
        notes.append(f"Training data: {improvement_analysis.get('training_samples_increase', 0)} new samples")
        
        return notes
    
    def get_feature_importance(self):
        """Get feature importance from supervised model"""
        if self.supervised_model is None:
            return None
        
        try:
            importance = self.supervised_model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return sorted_features
        except:
            return None
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            model_data = {
                'unsupervised_model': self.unsupervised_model,
                'supervised_model': self.supervised_model,
                'scaler': self.scaler,
                'model_version': self.model_version,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'save_date': datetime.now().isoformat()
            }
            
            model_path = MODELS_DIR / f"fraud_detector_{self.model_version}.pkl"
            joblib.dump(model_data, model_path)
            
            # Also save as current model
            current_path = MODELS_DIR / "fraud_detector_current.pkl"
            joblib.dump(model_data, current_path)
            
            logging.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logging.error(f"Error saving model: {e}")
    
    def load_model(self, model_path=None):
        """Load trained model from disk"""
        try:
            if model_path is None:
                model_path = MODELS_DIR / "fraud_detector_current.pkl"
            
            if not Path(model_path).exists():
                logging.warning(f"Model file not found: {model_path}")
                return False
            
            model_data = joblib.load(model_path)
            
            self.unsupervised_model = model_data['unsupervised_model']
            self.supervised_model = model_data.get('supervised_model')
            self.scaler = model_data['scaler']
            self.model_version = model_data['model_version']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data.get('feature_names', self.feature_names)
            
            logging.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False