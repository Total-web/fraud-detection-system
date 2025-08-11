import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime
import uuid

from core.database_manager import DatabaseManager
from core.data_processor import DataProcessor
from core.anomaly_detector import AnomalyDetector
from core.feedback_manager import FeedbackManager, RetrainingManager
from gui.upload_frame import UploadFrame
from gui.results_frame import ResultsFrame
from config.settings import UI_CONFIG

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.setup_logging()
        
        # Initialize core components
        self.db = DatabaseManager()
        self.data_processor = DataProcessor()
        self.detector = AnomalyDetector()
        self.feedback_manager = FeedbackManager(self.db)
        self.retraining_manager = RetrainingManager(self.db)
        
        # Session management
        self.current_session_id = None
        self.current_data = None
        self.current_filename = None
        self.analysis_results = None
        
        # GUI state
        self.current_frame = None
        
        # Setup UI
        self.setup_ui()
        self.setup_callbacks()
        
        # Try to load existing model
        model_loaded = self.detector.load_model()
        if model_loaded:
            self.model_info.config(text=f"Model: Loaded ({self.detector.model_version}) ✅")
        else:
            self.model_info.config(text="Model: Ready to train 🔄")
        
        logging.info("Application initialized successfully")
    
    def setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fraud_detection.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_ui(self):
        """Setup main user interface"""
        # Configure main window
        self.root.configure(bg=UI_CONFIG['colors']['background'])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.create_status_bar()
        
        # Create main content area
        self.create_main_content()
        
        # Show welcome screen initially
        self.show_upload_frame()
    
    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Upload CSV", command=self.show_upload_frame, accelerator="Ctrl+O")
        file_menu.add_command(label="Generate Sample Data", command=self.generate_sample_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results, accelerator="Ctrl+E")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="View Results", command=self.show_results_frame, accelerator="Ctrl+R")
        analysis_menu.add_command(label="Retrain Model", command=self.show_retraining_dialog, accelerator="Ctrl+T")
        
        # Investigation menu
        investigation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Investigation", menu=investigation_menu)
        investigation_menu.add_command(label="View Queue", command=self.show_investigation_queue, accelerator="Ctrl+I")
        investigation_menu.add_command(label="Investigation Reports", command=self.show_investigation_reports)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.show_upload_frame())
        self.root.bind('<Control-r>', lambda e: self.show_results_frame())
        self.root.bind('<Control-e>', lambda e: self.export_results())
        self.root.bind('<Control-t>', lambda e: self.show_retraining_dialog())
        self.root.bind('<Control-i>', lambda e: self.show_investigation_queue())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
    
    def create_toolbar(self):
        """Create application toolbar with investigation tracker"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(fill='x', padx=5, pady=5)
        
        # Left side buttons
        left_frame = ttk.Frame(self.toolbar)
        left_frame.pack(side='left')
        
        # Upload button
        self.upload_btn = ttk.Button(
            left_frame, 
            text="📁 Upload CSV", 
            command=self.show_upload_frame
        )
        self.upload_btn.pack(side='left', padx=5)
        
        # Analyze button (initially disabled)
        self.analyze_btn = ttk.Button(
            left_frame, 
            text="🔍 Analyze", 
            command=self.start_analysis,
            state='disabled'
        )
        self.analyze_btn.pack(side='left', padx=5)
        
        # Results button (initially disabled)
        self.results_btn = ttk.Button(
            left_frame, 
            text="📊 View Results", 
            command=self.show_results_frame,
            state='disabled'
        )
        self.results_btn.pack(side='left', padx=5)
        
        # Export button (initially disabled)
        self.export_btn = ttk.Button(
            left_frame, 
            text="💾 Export", 
            command=self.export_results,
            state='disabled'
        )
        self.export_btn.pack(side='left', padx=5)
        
        # Center - Model info
        center_frame = ttk.Frame(self.toolbar)
        center_frame.pack(side='left', expand=True, fill='x', padx=20)
        
        ttk.Separator(center_frame, orient='vertical').pack(side='left', fill='y', padx=10)
        
        self.model_info = ttk.Label(
            center_frame, 
            text="Model: Not Loaded",
            font=('Arial', 8)
        )
        self.model_info.pack(side='left', padx=5)
        
        # Right side - Investigation tracker and retrain button
        right_frame = ttk.Frame(self.toolbar)
        right_frame.pack(side='right')
        
        # Investigation tracker
        self.investigation_frame = ttk.LabelFrame(right_frame, text="🔍 Investigation Queue", padding=5)
        self.investigation_frame.pack(side='left', padx=5)
        
        self.investigation_count_label = ttk.Label(
            self.investigation_frame, 
            text="0 pending",
            font=('Arial', 8)
        )
        self.investigation_count_label.pack()
        
        self.view_investigations_btn = ttk.Button(
            self.investigation_frame,
            text="View Queue",
            command=self.show_investigation_queue,
            state='disabled'
        )
        self.view_investigations_btn.pack(pady=2)
        
        # Retrain button
        self.retrain_btn = ttk.Button(
            right_frame, 
            text="🤖 Retrain Model", 
            command=self.show_retraining_dialog,
            state='disabled'
        )
        self.retrain_btn.pack(side='right', padx=5)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill='x', side='bottom')
        
        # Status message
        self.status_label = ttk.Label(
            self.status_bar, 
            text="Ready to upload transaction data",
            relief='sunken'
        )
        self.status_label.pack(side='left', fill='x', expand=True, padx=5)
        
        # Progress bar (initially hidden)
        self.progress_bar = ttk.Progressbar(
            self.status_bar, 
            length=200, 
            mode='determinate'
        )
        
        # Session info
        self.session_label = ttk.Label(
            self.status_bar, 
            text="No active session",
            font=('Arial', 8)
        )
        self.session_label.pack(side='right', padx=5)
    
    def create_main_content(self):
        """Create main content area"""
        # Main content frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_callbacks(self):
        """Setup callbacks for feedback and retraining events"""
        try:
            # Feedback callbacks
            self.feedback_manager.register_callback('feedback_saved', self.on_feedback_saved)
            self.feedback_manager.register_callback('suggest_retraining', self.on_suggest_retraining)
            
            # Retraining callbacks
            self.retraining_manager.register_callback('retraining_started', self.on_retraining_started)
            self.retraining_manager.register_callback('retraining_progress', self.on_retraining_progress)
            self.retraining_manager.register_callback('retraining_completed', self.on_retraining_completed)
            self.retraining_manager.register_callback('retraining_error', self.on_retraining_error)
        except Exception as e:
            logging.error(f"Error setting up callbacks: {e}")
    
    def show_upload_frame(self):
        """Show file upload interface"""
        try:
            self.clear_main_frame()
            
            self.upload_frame = UploadFrame(self.main_frame, self)
            self.upload_frame.pack(fill='both', expand=True)
            
            self.current_frame = 'upload'
            self.update_status("Ready to upload transaction data")
        except Exception as e:
            logging.error(f"Error showing upload frame: {e}")
            messagebox.showerror("Error", f"Failed to show upload interface: {str(e)}")
    
    def show_results_frame(self):
        """Show analysis results interface"""
        if self.analysis_results is None:
            messagebox.showwarning("No Results", "Please analyze transaction data first.")
            return
        
        try:
            self.clear_main_frame()
            
            # Create results frame with error handling
            self.results_frame = ResultsFrame(self.main_frame, self, self.analysis_results)
            self.results_frame.pack(fill='both', expand=True)
            
            self.current_frame = 'results'
            self.update_status("Viewing analysis results")
            
            # Update investigation tracker (with error handling)
            try:
                self.update_investigation_tracker()
            except Exception as tracker_error:
                logging.error(f"Error updating investigation tracker: {tracker_error}")
                
        except Exception as e:
            logging.error(f"Error showing results frame: {e}")
            # Fallback - show simple results
            self.show_simple_results()
    
    def show_simple_results(self):
        """Show simple results as fallback"""
        try:
            self.clear_main_frame()
            
            # Create simple results display
            title_label = ttk.Label(
                self.main_frame,
                text="📊 Analysis Results",
                font=('Arial', 18, 'bold')
            )
            title_label.pack(pady=20)
            
            # Summary information
            flagged_count = sum(1 for r in self.analysis_results if r['ai_prediction'] == -1)
            normal_count = len(self.analysis_results) - flagged_count
            
            summary_text = f"""
Analysis Complete!

Total Transactions: {len(self.analysis_results)}
🚨 Flagged Transactions: {flagged_count}
✅ Normal Transactions: {normal_count}

Flagged transactions require review for potential fraud.
            """
            
            summary_label = ttk.Label(
                self.main_frame,
                text=summary_text,
                font=('Arial', 12),
                justify='center'
            )
            summary_label.pack(expand=True)
            
            # Create simple table for flagged transactions
            if flagged_count > 0:
                flagged_frame = ttk.LabelFrame(self.main_frame, text="🚨 Flagged Transactions", padding=10)
                flagged_frame.pack(fill='both', expand=True, padx=20, pady=20)
                
                # Simple treeview
                columns = ('ID', 'User', 'Amount', 'Risk Level', 'Confidence')
                tree = ttk.Treeview(flagged_frame, columns=columns, show='headings', height=10)
                
                for col in columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=120)
                
                # Add flagged transactions
                for result in self.analysis_results:
                    if result['ai_prediction'] == -1:
                        tree.insert('', 'end', values=(
                            result['transaction_id'],
                            result['user_id'],
                            f"Rs. {result['amount']:.2f}",
                            result['risk_level'].replace('_', ' ').title(),
                            f"{result['ai_confidence']:.1%}"
                        ))
                
                tree.pack(fill='both', expand=True)
                
                # Scrollbar
                scrollbar = ttk.Scrollbar(flagged_frame, orient='vertical', command=tree.yview)
                tree.configure(yscrollcommand=scrollbar.set)
                scrollbar.pack(side='right', fill='y')
            
            # Export button
            export_frame = ttk.Frame(self.main_frame)
            export_frame.pack(pady=10)
            
            ttk.Button(
                export_frame,
                text="💾 Export Results",
                command=self.export_results
            ).pack()
            
            self.update_status("Results displayed (simplified view)")
            
            # Also print to console as backup
            self.print_console_results()
            
        except Exception as e:
            logging.error(f"Error showing simple results: {e}")
            messagebox.showerror("Error", f"Failed to display results: {str(e)}")
    
    def print_console_results(self):
        """Print results to console as backup"""
        try:
            print(f"\n{'='*60}")
            print("FRAUD DETECTION RESULTS")
            print(f"{'='*60}")
            
            flagged_transactions = [r for r in self.analysis_results if r['ai_prediction'] == -1]
            normal_transactions = [r for r in self.analysis_results if r['ai_prediction'] == 1]
            
            print(f"Total Transactions: {len(self.analysis_results)}")
            print(f"Flagged Transactions: {len(flagged_transactions)}")
            print(f"Normal Transactions: {len(normal_transactions)}")
            print(f"{'='*60}")
            
            if flagged_transactions:
                print("\n🚨 FLAGGED TRANSACTIONS:")
                print("-" * 60)
                for result in flagged_transactions:
                    risk = result['risk_level'].replace('_', ' ').title()
                    confidence = f"{result['ai_confidence']:.1%}"
                    reasons = ', '.join(result['explanation']['reasons'][:2]) if result['explanation']['reasons'] else 'Unknown'
                    
                    print(f"ID: {result['transaction_id']} | User: {result['user_id']} | "
                          f"Amount: Rs.{result['amount']:.2f} | Risk: {risk} | "
                          f"Confidence: {confidence}")
                    print(f"  Reasons: {reasons}")
                    print()
            
            print(f"{'='*60}\n")
            
        except Exception as e:
            logging.error(f"Error printing console results: {e}")
    
    def clear_main_frame(self):
        """Clear main content frame"""
        try:
            for widget in self.main_frame.winfo_children():
                widget.destroy()
        except Exception as e:
            logging.error(f"Error clearing main frame: {e}")
    
    def start_analysis(self):
        """Start fraud detection analysis"""
        if self.current_data is None:
            messagebox.showerror("No Data", "Please upload transaction data first.")
            return
        
        try:
            self.update_status("Starting analysis...")
            self.show_progress_bar()
            self.progress_bar['value'] = 10
            self.root.update()
            
            # Create new session
            self.current_session_id = str(uuid.uuid4())
            
            self.progress_bar['value'] = 20
            self.root.update()
            
            # Build user profiles
            self.update_status("Building user profiles...")
            user_profiles = self.data_processor.build_user_profiles(self.current_data)
            
            self.progress_bar['value'] = 40
            self.root.update()
            
            # Extract features
            self.update_status("Extracting features...")
            all_features = []
            for _, transaction in self.current_data.iterrows():
                features = self.data_processor.extract_features(transaction, user_profiles)
                all_features.append(features)
            
            self.progress_bar['value'] = 60
            self.root.update()
            
            # Train unsupervised model if not trained
            if not self.detector.is_trained:
                self.update_status("Training anomaly detection model...")
                self.detector.train_unsupervised(all_features)
            
            self.progress_bar['value'] = 80
            self.root.update()
            
            # Detect anomalies
            self.update_status("Detecting anomalies...")
            predictions = self.detector.detect_anomalies(all_features)
            
            # Combine results
            results = []
            for i, (_, transaction) in enumerate(self.current_data.iterrows()):
                prediction = predictions[i]
                
                # Get explanation
                explanation = self.detector.explain_prediction(all_features[i], transaction)
                
                result = {
                    'transaction_id': transaction['TransactionID'],
                    'user_id': transaction['UserID'],
                    'amount': transaction['Amount'],
                    'timestamp': transaction['Time'],
                    'location': transaction['Location'],
                    'transaction_type': transaction['Type'],
                    'ai_prediction': -1 if prediction['is_anomaly'] else 1,
                    'ai_confidence': prediction['confidence'],
                    'risk_level': prediction['risk_level'],
                    'features': all_features[i],
                    'explanation': explanation,
                    'model_version': self.detector.model_version,
                    'needs_feedback': prediction['is_anomaly']  # Simple rule: only anomalies need feedback
                }
                results.append(result)
            
            self.progress_bar['value'] = 90
            self.root.update()
            
            # Save to database
            try:
                self.db.save_transaction_analysis(self.current_session_id, results)
                
                # Save session info
                flagged_count = sum(1 for r in results if r['ai_prediction'] == -1)
                self.db.save_session(
                    self.current_session_id,
                    getattr(self, 'current_filename', 'Unknown'),
                    len(results),
                    flagged_count
                )
            except Exception as db_error:
                logging.error(f"Database error: {db_error}")
                # Continue without database - analysis still works
            
            self.analysis_results = results
            
            self.progress_bar['value'] = 100
            self.root.update()
            
            # Update UI state
            self.results_btn['state'] = 'normal'
            self.export_btn['state'] = 'normal'
            self.retrain_btn['state'] = 'normal'
            
            # Update model info
            if self.detector.is_trained:
                model_type = "Supervised" if self.detector.supervised_model else "Unsupervised"
                self.model_info.config(text=f"Model: {model_type} ({self.detector.model_version}) ✅")
            
            # Show results
            self.hide_progress_bar()
            self.show_results_frame()
            
            # Show summary
            total = len(results)
            flagged = sum(1 for r in results if r['ai_prediction'] == -1)
            self.update_status(f"Analysis complete: {total} transactions, {flagged} flagged for review")
            
            # Update session info
            self.session_label.config(text=f"Session: {flagged}/{total} flagged")
            
            logging.info(f"Analysis completed: {total} transactions, {flagged} anomalies detected")
            
        except Exception as e:
            self.hide_progress_bar()
            self.update_status("Analysis failed")
            messagebox.showerror("Analysis Error", f"Error during analysis: {str(e)}")
            logging.error(f"Analysis error: {e}")
    
    def generate_sample_data(self):
        """Generate sample transaction data for demo"""
        try:
            self.update_status("Generating sample data...")
            
            # Generate sample data with error handling
            sample_data = self.data_processor.generate_sample_data(100)
            
            if sample_data is None or len(sample_data) == 0:
                raise ValueError("Failed to generate sample data")
            
            # Set as current data
            self.current_data = sample_data
            self.current_filename = "sample_data.csv"
            
            # Enable analyze button
            self.analyze_btn['state'] = 'normal'
            
            self.update_status(f"Sample data generated: {len(sample_data)} transactions")
            
            # Show notification
            messagebox.showinfo("Sample Data", 
                              f"Generated {len(sample_data)} sample transactions.\n"
                              "• 20 different users\n"
                              "• Mix of normal and suspicious patterns\n"
                              "• Ready for fraud detection analysis\n\n"
                              "Click 'Analyze' to start fraud detection.")
            
            logging.info(f"Generated {len(sample_data)} sample transactions")
            
        except Exception as e:
            error_msg = f"Failed to generate sample data: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.update_status("Sample data generation failed")
            logging.error(f"Sample data generation error: {e}")
            
            # Provide fallback option
            result = messagebox.askyesno("Use Fallback Data", 
                                       "Sample data generation failed.\n\n"
                                       "Would you like to use a smaller predefined dataset instead?\n"
                                       "This will contain 20 transactions with clear fraud patterns.")
            if result:
                try:
                    # Use the fallback data from data processor
                    fallback_data = self.data_processor._create_fallback_data()
                    self.current_data = fallback_data
                    self.current_filename = "fallback_data.csv"
                    self.analyze_btn['state'] = 'normal'
                    
                    self.update_status(f"Fallback data loaded: {len(fallback_data)} transactions")
                    messagebox.showinfo("Success", f"Loaded {len(fallback_data)} transactions with fraud patterns!")
                    
                except Exception as fallback_error:
                    messagebox.showerror("Error", f"Fallback data creation also failed: {str(fallback_error)}")
    
    def export_results(self):
        """Export analysis results"""
        if self.analysis_results is None:
            messagebox.showwarning("No Results", "No analysis results to export.")
            return
        
        try:
            from tkinter import filedialog
            import pandas as pd
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")],
                title="Export Results"
            )
            
            if not filename:
                return
            
            # Prepare export data
            export_data = []
            for result in self.analysis_results:
                export_row = {
                    'TransactionID': result['transaction_id'],
                    'UserID': result['user_id'],
                    'Amount': result['amount'],
                    'Time': result['timestamp'],
                    'Location': result['location'],
                    'Type': result['transaction_type'],
                    'RiskLevel': result['risk_level'],
                    'AIConfidence': round(result['ai_confidence'], 3),
                    'IsFlagged': 'Yes' if result['ai_prediction'] == -1 else 'No',
                    'Reasons': '; '.join(result['explanation']['reasons']) if result['explanation']['reasons'] else 'Normal pattern'
                }
                export_data.append(export_row)
            
            df = pd.DataFrame(export_data)
            
            # Export based on file extension
            if filename.endswith('.xlsx'):
                df.to_excel(filename, index=False)
            else:
                df.to_csv(filename, index=False)
            
            self.update_status(f"Results exported to {filename}")
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filename}")
            
            logging.info(f"Results exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
            logging.error(f"Export error: {e}")
    
    def show_retraining_dialog(self):
        """Show model retraining dialog"""
        try:
            can_retrain, message = self.retraining_manager.can_retrain()
            
            if not can_retrain:
                messagebox.showinfo("Cannot Retrain", message)
                return
            
            # Show confirmation dialog
            result = messagebox.askyesno(
                "Retrain Model",
                f"{message}\n\nThis will create a new supervised model using your feedback.\n"
                "Do you want to proceed?"
            )
            
            if result:
                self.retraining_manager.start_retraining()
        except Exception as e:
            logging.error(f"Error in retraining dialog: {e}")
            messagebox.showerror("Error", f"Retraining error: {str(e)}")
    
    def update_investigation_tracker(self):
        """Update the investigation tracker display"""
        try:
            if hasattr(self, 'results_frame') and hasattr(self.results_frame, 'investigation_flags'):
                pending_count = len([flag for flag in self.results_frame.investigation_flags if flag.get('status') == 'Pending'])
                
                self.investigation_count_label.config(text=f"{pending_count} pending")
                
                if pending_count > 0:
                    self.view_investigations_btn.config(state='normal')
                    self.investigation_frame.config(relief='raised', borderwidth=2)
                else:
                    self.view_investigations_btn.config(state='disabled')
                    self.investigation_frame.config(relief='flat', borderwidth=1)
            else:
                self.investigation_count_label.config(text="0 pending")
                self.view_investigations_btn.config(state='disabled')
        except Exception as e:
            logging.error(f"Error updating investigation tracker: {e}")
            self.investigation_count_label.config(text="0 pending")
            self.view_investigations_btn.config(state='disabled')
    
    def show_investigation_queue(self):
        """Show investigation queue window"""
        try:
            if not hasattr(self, 'results_frame') or not hasattr(self.results_frame, 'investigation_flags'):
                messagebox.showinfo("No Investigations", "No investigations currently flagged.\n\nPlease analyze transaction data and flag some transactions for investigation first.")
                return
            
            # Create investigation queue window
            queue_window = tk.Toplevel(self.root)
            queue_window.title("🔍 Investigation Queue")
            queue_window.geometry("900x600")
            queue_window.resizable(True, True)
            
            # Header
            header_frame = ttk.Frame(queue_window)
            header_frame.pack(fill='x', padx=10, pady=10)
            
            ttk.Label(header_frame, text="🔍 Investigation Queue", font=('Arial', 16, 'bold')).pack(side='left')
            
            # Refresh button
            refresh_btn = ttk.Button(header_frame, text="🔄 Refresh", command=lambda: self.refresh_investigation_queue(tree))
            refresh_btn.pack(side='right')
            
            # Investigation list
            list_frame = ttk.Frame(queue_window)
            list_frame.pack(fill='both', expand=True, padx=10, pady=5)
            
            # Create treeview for investigations
            columns = ('ID', 'Priority', 'Reason', 'Assigned', 'Date', 'Status')
            tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
            
            # Configure columns
            tree.heading('ID', text='Transaction ID')
            tree.heading('Priority', text='Priority')
            tree.heading('Reason', text='Reason')
            tree.heading('Assigned', text='Assigned To')
            tree.heading('Date', text='Date Flagged')
            tree.heading('Status', text='Status')
            
            tree.column('ID', width=100)
            tree.column('Priority', width=80)
            tree.column('Reason', width=200)
            tree.column('Assigned', width=100)
            tree.column('Date', width=120)
            tree.column('Status', width=80)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # Populate with investigation data
            self.populate_investigation_queue(tree)
            
            # Context menu for investigations
            def on_investigation_right_click(event):
                item = tree.identify_row(event.y)
                if item:
                    tree.selection_set(item)
                    context_menu = tk.Menu(queue_window, tearoff=0)
                    context_menu.add_command(label="Mark as Resolved", command=lambda: self.resolve_investigation(tree, item))
                    context_menu.add_command(label="Add Notes", command=lambda: self.add_investigation_notes(tree, item))
                    context_menu.add_command(label="View Transaction", command=lambda: self.view_investigation_transaction(tree, item))
                    
                    context_menu.post(event.x_root, event.y_root)
            
            tree.bind('<Button-3>', on_investigation_right_click)
            
            # Footer with summary
            footer_frame = ttk.Frame(queue_window)
            footer_frame.pack(fill='x', padx=10, pady=10)
            
            pending_count = len([flag for flag in self.results_frame.investigation_flags if flag.get('status') == 'Pending'])
            summary_label = ttk.Label(footer_frame, text=f"Total pending investigations: {pending_count}")
            summary_label.pack()
            
        except Exception as e:
            logging.error(f"Error showing investigation queue: {e}")
            messagebox.showerror("Error", f"Failed to show investigation queue: {str(e)}")
    
    def populate_investigation_queue(self, tree):
        """Populate investigation queue tree"""
        try:
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Add investigation flags
            if hasattr(self, 'results_frame') and hasattr(self.results_frame, 'investigation_flags'):
                for flag in self.results_frame.investigation_flags:
                    if flag.get('status') == 'Pending':
                        # Format date
                        try:
                            date_obj = datetime.fromisoformat(flag['flagged_date'])
                            formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
                        except:
                            formatted_date = flag.get('flagged_date', 'Unknown')
                        
                        # Truncate reason if too long
                        reason = flag.get('reason', 'No reason provided')
                        reason = reason[:50] + "..." if len(reason) > 50 else reason
                        
                        # Color code by priority
                        priority = flag.get('priority', 'Medium')
                        tags = []
                        if priority == 'Urgent':
                            tags = ['urgent']
                        elif priority == 'High':
                            tags = ['high']
                        elif priority == 'Medium':
                            tags = ['medium']
                        else:
                            tags = ['low']
                        
                        tree.insert('', 'end', values=(
                            flag.get('transaction_id', 'Unknown'),
                            priority,
                            reason,
                            flag.get('assigned_to', 'Unassigned'),
                            formatted_date,
                            flag.get('status', 'Pending')
                        ), tags=tags)
                
                # Configure tag colors
                tree.tag_configure('urgent', background='#ffcdd2', foreground='#b71c1c')
                tree.tag_configure('high', background='#ffe0b2', foreground='#e65100')
                tree.tag_configure('medium', background='#fff9c4', foreground='#f57f17')
                tree.tag_configure('low', background='#e8f5e8', foreground='#2e7d32')
                
        except Exception as e:
            logging.error(f"Error populating investigation queue: {e}")
    
    def refresh_investigation_queue(self, tree):
        """Refresh investigation queue"""
        try:
            if hasattr(self, 'results_frame'):
                self.results_frame.load_investigation_flags()
                self.populate_investigation_queue(tree)
                self.update_investigation_tracker()
        except Exception as e:
            logging.error(f"Error refreshing investigation queue: {e}")
    
    def resolve_investigation(self, tree, item):
        """Mark investigation as resolved"""
        try:
            values = tree.item(item)['values']
            transaction_id = values[0]
            
            # Show resolution dialog
            resolution_window = tk.Toplevel(self.root)
            resolution_window.title(f"Resolve Investigation - {transaction_id}")
            resolution_window.geometry("400x300")
            resolution_window.resizable(True, True)
            resolution_window.transient(self.root)
            resolution_window.grab_set()
            
            # Resolution notes
            ttk.Label(resolution_window, text="Resolution Notes:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10, 5))
            
            notes_text = tk.Text(resolution_window, height=8, wrap='word')
            notes_text.pack(fill='both', expand=True, padx=10, pady=5)
            
            # Buttons
            button_frame = ttk.Frame(resolution_window)
            button_frame.pack(fill='x', padx=10, pady=10)
            
            def save_resolution():
                notes = notes_text.get('1.0', 'end-1c').strip()
                if not notes:
                    messagebox.showwarning("Missing Notes", "Please provide resolution notes.")
                    return
                
                # Update database
                try:
                    import sqlite3
                    with sqlite3.connect(self.db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE investigation_flags 
                            SET status = 'Resolved', resolution_notes = ?, resolved_date = ?
                            WHERE transaction_id = ? AND status = 'Pending'
                        ''', (notes, datetime.now().isoformat(), transaction_id))
                        conn.commit()
                    
                    # Update local data
                    if hasattr(self, 'results_frame'):
                        for flag in self.results_frame.investigation_flags:
                            if flag.get('transaction_id') == transaction_id and flag.get('status') == 'Pending':
                                flag['status'] = 'Resolved'
                                flag['resolution_notes'] = notes
                                flag['resolved_date'] = datetime.now().isoformat()
                                break
                    
                    resolution_window.destroy()
                    self.refresh_investigation_queue(tree)
                    messagebox.showinfo("Resolved", f"Investigation for {transaction_id} marked as resolved.")
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to resolve investigation: {str(e)}")
                    logging.error(f"Error resolving investigation: {e}")
            
            ttk.Button(button_frame, text="Resolve", command=save_resolution).pack(side='left', padx=5)
            ttk.Button(button_frame, text="Cancel", command=resolution_window.destroy).pack(side='left', padx=5)
            
        except Exception as e:
            logging.error(f"Error in resolve investigation: {e}")
            messagebox.showerror("Error", f"Failed to resolve investigation: {str(e)}")
    
    def add_investigation_notes(self, tree, item):
        """Add notes to investigation"""
        try:
            values = tree.item(item)['values']
            transaction_id = values[0]
            
            # Create notes dialog
            notes_window = tk.Toplevel(self.root)
            notes_window.title(f"Add Notes - {transaction_id}")
            notes_window.geometry("400x250")
            notes_window.resizable(True, True)
            notes_window.transient(self.root)
            notes_window.grab_set()
            
            ttk.Label(notes_window, text="Investigation Notes:", font=('Arial', 10, 'bold')).pack(anchor='w', padx=10, pady=(10, 5))
            
            notes_text = tk.Text(notes_window, height=6, wrap='word')
            notes_text.pack(fill='both', expand=True, padx=10, pady=5)
            
            button_frame = ttk.Frame(notes_window)
            button_frame.pack(fill='x', padx=10, pady=10)
            
            def save_notes():
                notes = notes_text.get('1.0', 'end-1c').strip()
                if notes:
                    try:
                        # In a real implementation, save to database
                        messagebox.showinfo("Notes Saved", f"Notes saved for {transaction_id}")
                        notes_window.destroy()
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to save notes: {str(e)}")
                else:
                    messagebox.showwarning("Empty Notes", "Please enter some notes.")
            
            ttk.Button(button_frame, text="Save Notes", command=save_notes).pack(side='left', padx=5)
            ttk.Button(button_frame, text="Cancel", command=notes_window.destroy).pack(side='left', padx=5)
            
        except Exception as e:
            logging.error(f"Error adding investigation notes: {e}")
            messagebox.showerror("Error", f"Failed to add notes: {str(e)}")
    
    def view_investigation_transaction(self, tree, item):
        """View transaction details from investigation queue"""
        try:
            values = tree.item(item)['values']
            transaction_id = values[0]
            
            # Find transaction in results
            if hasattr(self, 'results_frame') and hasattr(self.results_frame, 'results'):
                result = next((r for r in self.results_frame.results if r['transaction_id'] == transaction_id), None)
                if result:
                    self.results_frame.show_simple_details(result)
                else:
                    messagebox.showwarning("Not Found", f"Transaction {transaction_id} not found in current results.")
            else:
                messagebox.showwarning("No Data", "No transaction data available.")
                
        except Exception as e:
            logging.error(f"Error viewing investigation transaction: {e}")
            messagebox.showerror("Error", f"Failed to view transaction: {str(e)}")
    
    def show_investigation_reports(self):
        """Show investigation reports"""
        try:
            # Create reports window
            reports_window = tk.Toplevel(self.root)
            reports_window.title("📊 Investigation Reports")
            reports_window.geometry("800x600")
            reports_window.resizable(True, True)
            
            # Header
            header_frame = ttk.Frame(reports_window)
            header_frame.pack(fill='x', padx=10, pady=10)
            
            ttk.Label(header_frame, text="📊 Investigation Reports", 
                     font=('Arial', 16, 'bold')).pack(side='left')
            
            # Content frame
            content_frame = ttk.Frame(reports_window)
            content_frame.pack(fill='both', expand=True, padx=10, pady=5)
            
            # Notebook for different report types
            notebook = ttk.Notebook(content_frame)
            notebook.pack(fill='both', expand=True)
            
            # Summary Report Tab
            summary_frame = ttk.Frame(notebook)
            notebook.add(summary_frame, text="Summary")
            self.create_summary_report(summary_frame)
            
            # Detailed Report Tab
            detailed_frame = ttk.Frame(notebook)
            notebook.add(detailed_frame, text="Detailed")
            self.create_detailed_report(detailed_frame)
            
            # Statistics Tab
            stats_frame = ttk.Frame(notebook)
            notebook.add(stats_frame, text="Statistics")
            self.create_statistics_report(stats_frame)
            
        except Exception as e:
            logging.error(f"Error showing investigation reports: {e}")
            messagebox.showerror("Error", f"Failed to show reports: {str(e)}")
    
    def create_summary_report(self, parent):
        """Create summary investigation report"""
        try:
            if not hasattr(self, 'results_frame') or not hasattr(self.results_frame, 'investigation_flags'):
                ttk.Label(parent, text="No investigation data available.", 
                         font=('Arial', 12)).pack(expand=True)
                return
            
            flags = self.results_frame.investigation_flags
            
            # Summary statistics
            stats_frame = ttk.LabelFrame(parent, text="Investigation Summary", padding=10)
            stats_frame.pack(fill='x', padx=10, pady=10)
            
            total_investigations = len(flags)
            pending_investigations = len([f for f in flags if f.get('status') == 'Pending'])
            resolved_investigations = len([f for f in flags if f.get('status') == 'Resolved'])
            
            # Priority breakdown
            priority_counts = {}
            for flag in flags:
                priority = flag.get('priority', 'Unknown')
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # Display stats
            stats_text = f"""Total Investigations: {total_investigations}
Pending: {pending_investigations}
Resolved: {resolved_investigations}

Priority Breakdown:
"""
            for priority, count in priority_counts.items():
                stats_text += f"• {priority}: {count}\n"
            
            if total_investigations > 0:
                resolution_rate = (resolved_investigations / total_investigations) * 100
                stats_text += f"\nResolution Rate: {resolution_rate:.1f}%"
            
            ttk.Label(stats_frame, text=stats_text, font=('Courier', 10)).pack(anchor='w')
            
            # Recent investigations
            recent_frame = ttk.LabelFrame(parent, text="Recent Investigations", padding=10)
            recent_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Create treeview for recent investigations
            columns = ('ID', 'Priority', 'Status', 'Date')
            tree = ttk.Treeview(recent_frame, columns=columns, show='headings', height=10)
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150)
            
            # Sort flags by date (most recent first)
            sorted_flags = sorted(flags, key=lambda x: x.get('flagged_date', ''), reverse=True)
            
            for flag in sorted_flags[:10]:  # Show last 10
                try:
                    date_obj = datetime.fromisoformat(flag.get('flagged_date', ''))
                    formatted_date = date_obj.strftime('%Y-%m-%d')
                except:
                    formatted_date = flag.get('flagged_date', 'Unknown')
                
                tree.insert('', 'end', values=(
                    flag.get('transaction_id', 'Unknown'),
                    flag.get('priority', 'Unknown'),
                    flag.get('status', 'Unknown'),
                    formatted_date
                ))
            
            tree.pack(fill='both', expand=True)
            
        except Exception as e:
            logging.error(f"Error creating summary report: {e}")
            ttk.Label(parent, text=f"Error generating summary report: {str(e)}", 
                     font=('Arial', 10), foreground='red').pack(expand=True)
    
    def create_detailed_report(self, parent):
        """Create detailed investigation report"""
        try:
            if not hasattr(self, 'results_frame') or not hasattr(self.results_frame, 'investigation_flags'):
                ttk.Label(parent, text="No investigation data available.", 
                         font=('Arial', 12)).pack(expand=True)
                return
            
            # Create detailed list with all investigations
            detail_frame = ttk.Frame(parent)
            detail_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Create treeview for detailed view
            columns = ('ID', 'Priority', 'Reason', 'Assigned', 'Status', 'Date')
            tree = ttk.Treeview(detail_frame, columns=columns, show='headings', height=20)
            
            # Configure columns
            column_widths = {'ID': 100, 'Priority': 80, 'Reason': 200, 'Assigned': 100, 
                            'Status': 80, 'Date': 100}
            
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=column_widths.get(col, 100))
            
            # Populate with all investigations
            for flag in self.results_frame.investigation_flags:
                try:
                    flagged_date = datetime.fromisoformat(flag.get('flagged_date', '')).strftime('%Y-%m-%d')
                except:
                    flagged_date = flag.get('flagged_date', 'Unknown')
                
                reason = flag.get('reason', '')[:50] + '...' if len(flag.get('reason', '')) > 50 else flag.get('reason', '')
                
                tree.insert('', 'end', values=(
                    flag.get('transaction_id', 'Unknown'),
                    flag.get('priority', 'Unknown'),
                    reason,
                    flag.get('assigned_to', 'Unassigned'),
                    flag.get('status', 'Unknown'),
                    flagged_date
                ))
            
            tree.pack(side='left', fill='both', expand=True)
            
            # Scrollbar
            scrollbar = ttk.Scrollbar(detail_frame, orient='vertical', command=tree.yview)
            scrollbar.pack(side='right', fill='y')
            tree.configure(yscrollcommand=scrollbar.set)
            
            # Export button
            export_frame = ttk.Frame(parent)
            export_frame.pack(fill='x', padx=10, pady=10)
            
            ttk.Button(export_frame, text="📤 Export Detailed Report", 
                      command=self.export_investigation_report).pack(side='right')
            
        except Exception as e:
            logging.error(f"Error creating detailed report: {e}")
            ttk.Label(parent, text=f"Error generating detailed report: {str(e)}", 
                     font=('Arial', 10), foreground='red').pack(expand=True)
    
    def create_statistics_report(self, parent):
        """Create statistics investigation report"""
        try:
            if not hasattr(self, 'results_frame') or not hasattr(self.results_frame, 'investigation_flags'):
                ttk.Label(parent, text="No investigation data available.", 
                         font=('Arial', 12)).pack(expand=True)
                return
            
            flags = self.results_frame.investigation_flags
            
            # Statistics content
            stats_content = ttk.Frame(parent)
            stats_content.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Create text widget for statistics
            text_widget = tk.Text(stats_content, wrap='word', font=('Courier', 10))
            text_widget.pack(fill='both', expand=True)
            
            # Generate statistics
            stats_text = self.generate_investigation_statistics(flags)
            text_widget.insert('1.0', stats_text)
            text_widget.config(state='disabled')
            
            # Scrollbar for text
            scrollbar = ttk.Scrollbar(stats_content, orient='vertical', command=text_widget.yview)
            scrollbar.pack(side='right', fill='y')
            text_widget.configure(yscrollcommand=scrollbar.set)
            
        except Exception as e:
            logging.error(f"Error creating statistics report: {e}")
            ttk.Label(parent, text=f"Error generating statistics: {str(e)}", 
                     font=('Arial', 10), foreground='red').pack(expand=True)
    
    def generate_investigation_statistics(self, flags):
        """Generate detailed investigation statistics"""
        try:
            if not flags:
                return "No investigation data available."
            
            total = len(flags)
            pending = len([f for f in flags if f.get('status') == 'Pending'])
            resolved = len([f for f in flags if f.get('status') == 'Resolved'])
            
            # Priority analysis
            priority_stats = {}
            for flag in flags:
                priority = flag.get('priority', 'Unknown')
                priority_stats[priority] = priority_stats.get(priority, 0) + 1
            
            # Assignment analysis
            assignment_stats = {}
            for flag in flags:
                assigned = flag.get('assigned_to', 'Unassigned')
                assignment_stats[assigned] = assignment_stats.get(assigned, 0) + 1
            
            # Generate report
            report = f"""
INVESTIGATION STATISTICS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

========================================
OVERALL SUMMARY
========================================
Total Investigations: {total}
Pending: {pending} ({pending/total*100:.1f}%)
Resolved: {resolved} ({resolved/total*100:.1f}%)

========================================
PRIORITY BREAKDOWN
========================================
"""
            
            for priority, count in sorted(priority_stats.items()):
                percentage = (count / total) * 100
                report += f"{priority}: {count} ({percentage:.1f}%)\n"
            
            report += f"""
========================================
ASSIGNMENT ANALYSIS
========================================
"""
            
            for assignee, count in sorted(assignment_stats.items()):
                percentage = (count / total) * 100
                report += f"{assignee}: {count} ({percentage:.1f}%)\n"
            
            report += f"""
========================================
RECENT ACTIVITY
========================================
"""
            
            # Sort by date and show recent activity
            sorted_flags = sorted(flags, key=lambda x: x.get('flagged_date', ''), reverse=True)
            
            for flag in sorted_flags[:5]:  # Show last 5
                try:
                    date_obj = datetime.fromisoformat(flag.get('flagged_date', ''))
                    formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
                except:
                    formatted_date = flag.get('flagged_date', 'Unknown')
                
                status_icon = "✅" if flag.get('status') == 'Resolved' else "⏳"
                report += f"{status_icon} {flag.get('transaction_id', 'Unknown')} - {flag.get('priority', 'Unknown')} - {formatted_date}\n"
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating statistics: {e}")
            return f"Error generating statistics: {str(e)}"
    
    def export_investigation_report(self):
        """Export investigation report to CSV"""
        try:
            from tkinter import filedialog
            import pandas as pd
            
            if not hasattr(self, 'results_frame') or not hasattr(self.results_frame, 'investigation_flags'):
                messagebox.showwarning("No Data", "No investigation data to export.")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")],
                title="Export Investigation Report"
            )
            
            if not filename:
                return
            
            # Prepare export data
            export_data = []
            for flag in self.results_frame.investigation_flags:
                export_row = {
                    'TransactionID': flag.get('transaction_id', 'Unknown'),
                    'Priority': flag.get('priority', 'Unknown'),
                    'Reason': flag.get('reason', ''),
                    'AssignedTo': flag.get('assigned_to', 'Unassigned'),
                    'Status': flag.get('status', 'Unknown'),
                    'FlaggedDate': flag.get('flagged_date', ''),
                    'ResolvedDate': flag.get('resolved_date', ''),
                    'ResolutionNotes': flag.get('resolution_notes', '')
                }
                export_data.append(export_row)
            
            df = pd.DataFrame(export_data)
            
            # Export based on file extension
            if filename.endswith('.xlsx'):
                df.to_excel(filename, index=False)
            else:
                df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Investigation report exported to:\n{filename}")
            logging.info(f"Investigation report exported: {len(export_data)} investigations")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report: {str(e)}")
            logging.error(f"Investigation report export error: {e}")
    
    def show_progress_bar(self):
        """Show progress bar in status bar"""
        self.progress_bar.pack(side='right', padx=5)
        self.progress_bar['value'] = 0
    
    def hide_progress_bar(self):
        """Hide progress bar"""
        self.progress_bar.pack_forget()
    
    def update_status(self, message):
        """Update status bar message"""
        try:
            self.status_label.config(text=message)
            self.root.update_idletasks()
        except Exception as e:
            logging.error(f"Error updating status: {e}")
    
    # Callback methods for feedback and retraining events
    def on_feedback_saved(self, feedback_data):
        """Handle feedback saved event"""
        try:
            # Update UI to show feedback was saved
            if hasattr(self, 'results_frame'):
                self.results_frame.update_feedback_display(
                    feedback_data['transaction_id'],
                    feedback_data['feedback']
                )
        except Exception as e:
            logging.error(f"Error handling feedback saved: {e}")
    
    def on_suggest_retraining(self, stats):
        """Handle retraining suggestion"""
        try:
            # Show subtle notification
            self.show_retraining_notification(stats)
        except Exception as e:
            logging.error(f"Error showing retraining suggestion: {e}")
    
    def on_retraining_started(self, data):
        """Handle retraining started event"""
        try:
            self.update_status(data['status'])
            self.show_progress_bar()
            self.retrain_btn['state'] = 'disabled'
        except Exception as e:
            logging.error(f"Error handling retraining started: {e}")
    
    def on_retraining_progress(self, data):
        """Handle retraining progress update"""
        try:
            self.update_status(data['status'])
            self.progress_bar['value'] = data['progress']
            self.root.update_idletasks()
        except Exception as e:
            logging.error(f"Error handling retraining progress: {e}")
    
    def on_retraining_completed(self, data):
        """Handle retraining completion with enhanced feedback"""
        try:
            self.hide_progress_bar()
            self.retrain_btn['state'] = 'normal'
            
            if data['success']:
                performance = data['performance']
                improvement_analysis = performance.get('improvement_analysis', {})
                
                # Create detailed success message
                message_parts = [f"🎉 Model retrained successfully!"]
                message_parts.append(f"New model version: {data['model_version']}")
                message_parts.append("")
                message_parts.append("📊 Performance Metrics:")
                message_parts.append(f"• Accuracy: {performance['accuracy']:.1%}")
                message_parts.append(f"• Precision: {performance['precision']:.1%}")
                message_parts.append(f"• Recall: {performance['recall']:.1%}")
                message_parts.append(f"• F1-Score: {performance['f1']:.1%}")
                
                if not improvement_analysis.get('is_first_model', True):
                    message_parts.append("")
                    if improvement_analysis.get('improvements'):
                        message_parts.append("✅ Improvements:")
                        for improvement in improvement_analysis.get('improvements', []):
                            message_parts.append(f"   {improvement}")
                    
                    if improvement_analysis.get('regressions'):
                        message_parts.append("")
                        message_parts.append("⚠️ Areas to monitor:")
                        for regression in improvement_analysis.get('regressions', []):
                            message_parts.append(f"   {regression}")
                    
                    message_parts.append("")
                    message_parts.append(f"📈 Overall: {improvement_analysis.get('summary', 'Model updated')}")
                else:
                    message_parts.append("")
                    message_parts.append("🎯 This is your first supervised model!")
                    message_parts.append("Continue providing feedback to improve accuracy.")
                
                message = "\n".join(message_parts)
                
                # Create custom dialog for better display
                result_dialog = tk.Toplevel(self.root)
                result_dialog.title("Model Learning Complete")
                result_dialog.geometry("500x400")
                result_dialog.resizable(False, False)
                
                # Make modal
                result_dialog.transient(self.root)
                result_dialog.grab_set()
                
                # Header
                header_frame = ttk.Frame(result_dialog)
                header_frame.pack(fill='x', padx=20, pady=10)
                
                ttk.Label(header_frame, text="🤖 Model Learning Complete", 
                         font=('Arial', 14, 'bold')).pack()
                
                # Content
                content_frame = ttk.Frame(result_dialog)
                content_frame.pack(fill='both', expand=True, padx=20, pady=10)
                
                # Use text widget for better formatting
                text_widget = tk.Text(content_frame, wrap='word', height=15, width=60)
                text_widget.pack(fill='both', expand=True)
                text_widget.insert('1.0', message)
                text_widget.config(state='disabled')
                
                # Add scrollbar
                scrollbar = ttk.Scrollbar(content_frame, orient='vertical', command=text_widget.yview)
                scrollbar.pack(side='right', fill='y')
                text_widget.config(yscrollcommand=scrollbar.set)
                
                # Close button
                ttk.Button(result_dialog, text="Continue", 
                          command=result_dialog.destroy).pack(pady=10)
                
                self.update_status("Model successfully learned from feedback")
                
                # Update model info with enhanced version
                version_type = "Enhanced" if improvement_analysis.get('overall_improvement') else "Experimental"
                self.model_info.config(text=f"Model: {version_type} ({data['model_version']}) ✅")
                
            else:
                messagebox.showerror("Retraining Failed", f"Model retraining failed:\n{data['error']}")
                self.update_status("Model retraining failed")
        except Exception as e:
            logging.error(f"Error handling retraining completion: {e}")
    
    def on_retraining_error(self, data):
        """Handle retraining error"""
        try:
            self.hide_progress_bar()
            self.retrain_btn['state'] = 'normal'
            messagebox.showerror("Retraining Error", data['message'])
            self.update_status("Retraining error occurred")
        except Exception as e:
            logging.error(f"Error handling retraining error: {e}")
    
    def show_retraining_notification(self, stats):
        """Show subtle retraining notification"""
        try:
            # Create notification frame
            notification = ttk.Frame(self.status_bar, relief='raised', borderwidth=1)
            
            message = ttk.Label(
                notification,
                text=f"💡 Ready to improve accuracy with {stats['total_reviewed']} feedback samples",
                foreground='blue',
                font=('Arial', 8)
            )
            
            retrain_btn = ttk.Button(
                notification,
                text="Retrain Now",
                command=lambda: [notification.destroy(), self.show_retraining_dialog()]
            )
            
            dismiss_btn = ttk.Button(
                notification,
                text="Later",
                command=notification.destroy
            )
            
            message.pack(side='left', padx=5)
            retrain_btn.pack(side='left', padx=2)
            dismiss_btn.pack(side='left', padx=2)
            
            notification.pack(side='right', padx=10, pady=2)
            
            # Auto-dismiss after 30 seconds
            self.root.after(30000, lambda: notification.destroy() if notification.winfo_exists() else None)
        except Exception as e:
            logging.error(f"Error showing retraining notification: {e}")
    
    def show_help(self):
        """Show help dialog"""
        try:
            help_text = """
Smart Transaction Monitor - User Guide

1. Upload CSV File:
  - Click 'Upload CSV' and select your transaction file
  - Required columns: TransactionID, UserID, Amount, Time, Location, Type
  - Or use 'Generate Sample Data' for demo

2. Analyze Transactions:
  - Click 'Analyze' to start fraud detection
  - System will build user profiles and detect anomalies
  - Progress bar shows analysis status

3. Review Results:
  - Flagged transactions will be highlighted by risk level
  - Right-click for feedback options: Mark as Legitimate/Fraud or Flag for Investigation
  - Double-click for detailed explanations
  - Use filters to focus on specific risk levels or users

4. Provide Feedback:
  - Mark transactions as legitimate or fraud (with risk levels)
  - System learns from your feedback to improve accuracy
  - Feedback progress is tracked at the bottom

5. Flag for Investigation:
  - For uncertain cases that need manual review
  - Select reason and priority level (Low, Medium, High, Urgent)
  - Assign to team members for investigation
  - Track investigations in the Investigation Queue

6. Investigation Queue:
  - View all flagged investigations with priority-based coloring
  - Right-click to mark as resolved, add notes, or view transaction
  - Filter by priority and assignment status
  - Track investigation progress and resolution

7. Investigation Reports:
  - Summary reports with statistics and recent activity
  - Detailed reports with all investigation data
  - Statistics analysis with resolution times and breakdowns
  - Export investigation reports as CSV or Excel

8. Improve the Model:
  - System learns from your feedback automatically
  - Retrain model when prompted for better accuracy
  - New versions show performance improvements and comparisons
  - Model versioning tracks improvements over time

9. Export Results:
  - Save analysis results as CSV or Excel file
  - Include user feedback and investigation status
  - Export filtered results or investigation reports

Keyboard Shortcuts:
- Ctrl+O: Upload file
- Ctrl+R: View results
- Ctrl+E: Export results
- Ctrl+T: Retrain model
- Ctrl+I: Investigation queue
- Ctrl+Q: Quit application

Tips:
- Use sample data generation for quick testing
- Provide feedback on flagged transactions to improve accuracy
- Flag uncertain cases for investigation rather than guessing
- Review investigation reports regularly to track team performance
- Export data for external analysis or record keeping

Investigation Features:
- Priority levels with color coding: Low (green), Medium (yellow), High (orange), Urgent (red)
- Assignment to team members with tracking
- Resolution workflow with detailed notes
- Real-time queue updates and status tracking
- Comprehensive reporting system with statistics
- Export capabilities for audit trails

The system continuously learns and improves with your feedback!
           """
            
            help_window = tk.Toplevel(self.root)
            help_window.title("User Guide - Smart Transaction Monitor")
            help_window.geometry("800x700")
            help_window.resizable(True, True)
            
            # Create main frame with scrollbar
            main_frame = ttk.Frame(help_window)
            main_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(main_frame, wrap='word', font=('Arial', 10))
            text_widget.pack(side='left', fill='both', expand=True)
            text_widget.insert('1.0', help_text)
            text_widget.config(state='disabled')
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=text_widget.yview)
            scrollbar.pack(side='right', fill='y')
            text_widget.config(yscrollcommand=scrollbar.set)
            
            # Close button
            button_frame = ttk.Frame(help_window)
            button_frame.pack(fill='x', padx=10, pady=10)
            ttk.Button(button_frame, text="Close", command=help_window.destroy).pack()
            
        except Exception as e:
            logging.error(f"Error showing help: {e}")
            messagebox.showerror("Error", f"Failed to show help: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        try:
            about_text = """
Smart Transaction Monitor
AI-Powered Fraud Detection System

Version: 2.1 Enhanced
Built with Python & Tkinter

Core Features:
- Advanced anomaly detection with Machine Learning
- Smart model learning from user feedback
- Real-time fraud scoring with explainable AI
- Investigation flagging and tracking system
- Intelligent model versioning and improvement tracking
- Team collaboration tools for investigations

Investigation Management:
- Priority-based investigation queue (Low, Medium, High, Urgent)
- Assignment and tracking system for team members
- Resolution workflow with detailed notes and audit trail
- Real-time status updates and notifications
- Comprehensive reporting system with statistics
- Export capabilities for compliance and record keeping

Technical Specifications:
- Isolation Forest for unsupervised anomaly detection
- Random Forest for supervised learning with feedback
- Feature engineering for transaction behavior analysis
- SQLite database for persistent storage
- Responsive GUI with professional styling
- Cross-platform compatibility (Windows, Linux, macOS)

Machine Learning Features:
- User behavior profiling and baseline establishment
- Multi-dimensional feature extraction from transactions
- Adaptive learning from human feedback
- Model performance tracking and comparison
- Automatic retraining suggestions based on feedback volume
- Version control for model improvements

Security & Compliance:
- Local data processing (no external data transmission)
- Audit trail for all investigations and decisions
- Export capabilities for regulatory compliance
- User feedback tracking for model accountability
- Detailed logging for system monitoring

Performance Metrics:
- Real-time accuracy measurements
- Precision and recall tracking
- False positive/negative analysis
- Investigation resolution time tracking
- Team performance statistics

Developed for banking and financial security applications.
Suitable for transaction monitoring, fraud prevention, and compliance.

© 2024 - Fraud Detection Research Project
Contact: support@frauddetection.ai
           """
            
            about_window = tk.Toplevel(self.root)
            about_window.title("About Smart Transaction Monitor")
            about_window.geometry("600x500")
            about_window.resizable(True, True)
            
            # Make modal
            about_window.transient(self.root)
            about_window.grab_set()
            
            # Header with icon
            header_frame = ttk.Frame(about_window)
            header_frame.pack(fill='x', padx=20, pady=20)
            
            ttk.Label(header_frame, text="🔍🤖", font=('Arial', 32)).pack()
            ttk.Label(header_frame, text="Smart Transaction Monitor", 
                     font=('Arial', 18, 'bold')).pack(pady=5)
            ttk.Label(header_frame, text="AI-Powered Fraud Detection System", 
                     font=('Arial', 12, 'italic')).pack()
            
            # Content with scrollbar
            content_frame = ttk.Frame(about_window)
            content_frame.pack(fill='both', expand=True, padx=20)
            
            text_widget = tk.Text(content_frame, wrap='word', font=('Arial', 9))
            text_widget.pack(side='left', fill='both', expand=True)
            text_widget.insert('1.0', about_text)
            text_widget.config(state='disabled')
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(content_frame, orient='vertical', command=text_widget.yview)
            scrollbar.pack(side='right', fill='y')
            text_widget.config(yscrollcommand=scrollbar.set)
            
            # Close button
            button_frame = ttk.Frame(about_window)
            button_frame.pack(fill='x', padx=20, pady=20)
            
            ttk.Button(button_frame, text="Close", command=about_window.destroy).pack()
            
        except Exception as e:
            logging.error(f"Error showing about: {e}")
            messagebox.showerror("Error", f"Failed to show about dialog: {str(e)}")