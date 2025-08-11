import queue
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging
import sqlite3
from core.database_manager import DatabaseManager

class FeedbackManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.feedback_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        self.callbacks = {}  # UI update callbacks
        
        self.start_background_processor()
    
    def start_background_processor(self):
        """Start background thread for processing feedback"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._process_feedback_queue,
            daemon=True,
            name="FeedbackProcessor"
        )
        self.processing_thread.start()
        logging.info("Feedback processor started")
    
    def stop_background_processor(self):
        """Stop background processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
    
    def add_feedback(self, transaction_id: str, feedback: str, feedback_type: str = 'quick'):
        """Add feedback to processing queue (non-blocking)"""
        feedback_data = {
            'transaction_id': transaction_id,
            'feedback': feedback,
            'feedback_type': feedback_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_queue.put(feedback_data)
        logging.info(f"Feedback queued for transaction {transaction_id}: {feedback}")
    
    def _process_feedback_queue(self):
        """Background processing of feedback"""
        while self.is_running:
            try:
                # Get feedback from queue with timeout
                feedback = self.feedback_queue.get(timeout=1)
                
                # Save to database
                success = self.db.save_feedback(
                    feedback['transaction_id'],
                    feedback['feedback'],
                    feedback['feedback_type']
                )
                
                if success:
                    # Notify UI of successful feedback
                    self._notify_ui('feedback_saved', feedback)
                    
                    # Check if we should suggest retraining
                    self._check_retraining_threshold()
                else:
                    logging.error(f"Failed to save feedback for {feedback['transaction_id']}")
                
                self.feedback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing feedback: {e}")
    
    def register_callback(self, event_type: str, callback):
        """Register UI callback for events"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def _notify_ui(self, event_type: str, data):
        """Notify UI components of events"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logging.error(f"Error in UI callback: {e}")
    
    def _check_retraining_threshold(self):
        """Check if we have enough feedback for retraining"""
        stats = self.db.get_feedback_stats()
        
        # Check if we have enough new feedback
        if stats['total_reviewed'] >= 10 and stats['total_reviewed'] % 10 == 0:
            # Suggest retraining every 10 feedback items
            self._notify_ui('suggest_retraining', stats)
    
    def get_feedback_summary(self, session_id: Optional[str] = None) -> Dict:
        """Get summary of feedback for current session or all time"""
        return self.db.get_feedback_stats(session_id)
    
    def should_request_feedback(self, prediction_result: Dict, transaction_data: Dict) -> tuple:
        """Smart logic to decide when to ask for feedback"""
        confidence = prediction_result.get('confidence', 0.5)
        is_anomaly = prediction_result.get('is_anomaly', False)
        
        # Always skip normal transactions with high confidence
        if not is_anomaly and confidence > 0.9:
            return False, "high_confidence_normal"
        
        # Always request feedback for high-risk anomalies
        if is_anomaly and confidence > 0.8:
            return True, "high_confidence_anomaly"
        
        # Request feedback for uncertain predictions
        if 0.3 < confidence < 0.7:
            return True, "uncertain_prediction"
        
        # Randomly sample some confident predictions (10%)
        import random
        if random.random() < 0.1:
            return True, "validation_sample"
        
        return False, "skipped"

class RetrainingManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.detector = None  # Will be set when needed
        self.retraining_in_progress = False
        self.last_retrain_date = None
        self.callbacks = {}
    
    def register_callback(self, event_type: str, callback):
        """Register callback for retraining events"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def _notify_ui(self, event_type: str, data):
        """Notify UI of retraining events"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logging.error(f"Error in retraining callback: {e}")
    
    def can_retrain(self) -> tuple:
        """Check if retraining is possible"""
        try:
            features, labels = self.db.get_training_data()
            
            if features is None or len(features) < 10:
                return False, "Need at least 10 feedback samples for retraining"
            
            if self.retraining_in_progress:
                return False, "Retraining already in progress"
            
            return True, f"Ready to retrain with {len(features)} samples"
            
        except Exception as e:
            logging.error(f"Error checking retrain capability: {e}")
            return False, f"Error checking retraining capability: {str(e)}"
    
    def start_retraining(self) -> bool:
        """Start retraining process in background"""
        can_retrain, message = self.can_retrain()
        if not can_retrain:
            self._notify_ui('retraining_error', {'message': message})
            return False
        
        # Start retraining in background thread
        threading.Thread(
            target=self._retrain_worker,
            daemon=True,
            name="RetrainingWorker"
        ).start()
        
        return True
    
    def _retrain_worker(self):
        """Background retraining worker"""
        try:
            self.retraining_in_progress = True
            self._notify_ui('retraining_started', {'status': 'Preparing training data...'})
            
            # Get training data
            features, labels = self.db.get_training_data()
            
            if features is None:
                raise ValueError("No training data available")
            
            self._notify_ui('retraining_progress', {'status': 'Training model...', 'progress': 30})
            
            # Import detector here to avoid circular imports
            from core.anomaly_detector import AnomalyDetector
            self.detector = AnomalyDetector()
            
            # Load existing unsupervised model as base
            self.detector.load_model()
            
            self._notify_ui('retraining_progress', {'status': 'Training supervised model...', 'progress': 60})
            
            # Train supervised model
            success, performance = self.detector.train_supervised(features, labels)
            
            if success:
                self._notify_ui('retraining_progress', {'status': 'Validating model...', 'progress': 90})
                
                # Save performance metrics
                self._save_performance_metrics(performance)
                
                self._notify_ui('retraining_completed', {
                    'success': True,
                    'performance': performance,
                    'model_version': self.detector.model_version
                })
                
                self.last_retrain_date = datetime.now()
                logging.info(f"Retraining completed successfully: {performance}")
            else:
                raise ValueError("Model training failed")
                
        except Exception as e:
            logging.error(f"Retraining failed: {e}")
            self._notify_ui('retraining_completed', {
                'success': False,
                'error': str(e)
            })
        finally:
            self.retraining_in_progress = False
    
    def _save_performance_metrics(self, performance: Dict):
        """Save model performance to database"""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO model_performance 
                    (model_version, accuracy, precision_score, recall_score, 
                     f1_score, training_date, samples_count, feedback_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.detector.model_version,
                    performance['accuracy'],
                    performance['precision'],
                    performance['recall'],
                    performance['f1'],
                    datetime.now().isoformat(),
                    performance['training_samples'],
                    performance['training_samples']
                ))
                conn.commit()
                logging.info("Performance metrics saved successfully")
                
        except Exception as e:
            logging.error(f"Error saving performance metrics: {e}")