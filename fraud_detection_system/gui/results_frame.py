import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime
import logging

class ResultsFrame(ttk.Frame):
    def __init__(self, parent, main_window, results):
        super().__init__(parent)
        self.main_window = main_window
        self.results = results
        self.filtered_results = results.copy()
        self.feedback_widgets = {}
        self.investigation_flags = []  # Track investigation flags
        
        # Statistics
        self.total_transactions = len(results)
        self.flagged_transactions = len([r for r in results if r['ai_prediction'] == -1])
        self.feedback_count = 0
        
        self.setup_ui()
        self.populate_results()
        self.load_investigation_flags()
    
    def setup_ui(self):
        """Setup results interface"""
        # Header with summary
        self.create_header()
        
        # Filter and control panel
        self.create_filter_panel()
        
        # Results table
        self.create_results_table()
        
        # Footer with actions
        self.create_footer()
    
    def create_header(self):
        """Create header with analysis summary"""
        header_frame = ttk.Frame(self)
        header_frame.pack(fill='x', padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text="🔍 Fraud Detection Results",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(side='left')
        
        # Summary stats
        stats_frame = ttk.Frame(header_frame)
        stats_frame.pack(side='right')
        
        # Create summary cards
        self.create_summary_card(stats_frame, "Total", self.total_transactions, 'blue')
        self.create_summary_card(stats_frame, "Flagged", self.flagged_transactions, 'red')
        normal_count = self.total_transactions - self.flagged_transactions
        self.create_summary_card(stats_frame, "Normal", normal_count, 'green')
    
    def create_summary_card(self, parent, label, value, color):
        """Create a summary statistics card"""
        card_frame = ttk.Frame(parent, relief='raised', borderwidth=1)
        card_frame.pack(side='left', padx=5)
        
        value_label = ttk.Label(
            card_frame,
            text=str(value),
            font=('Arial', 18, 'bold'),
            foreground=color
        )
        value_label.pack(padx=10, pady=2)
        
        label_label = ttk.Label(
            card_frame,
            text=label,
            font=('Arial', 8)
        )
        label_label.pack(padx=10, pady=2)
    
    def create_filter_panel(self):
        """Create filter and control panel"""
        filter_frame = ttk.LabelFrame(self, text="Filters & Controls", padding=10)
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        # Risk level filter
        ttk.Label(filter_frame, text="Risk Level:").pack(side='left', padx=5)
        
        self.risk_filter = ttk.Combobox(
            filter_frame,
            values=['All', 'High Risk', 'Suspicious', 'Low Risk', 'Normal'],
            state='readonly',
            width=12
        )
        self.risk_filter.set('All')
        self.risk_filter.pack(side='left', padx=5)
        self.risk_filter.bind('<<ComboboxSelected>>', self.apply_filters)
        
        # User filter
        ttk.Label(filter_frame, text="User:").pack(side='left', padx=(20, 5))
        
        user_ids = sorted(list(set(r['user_id'] for r in self.results)))
        self.user_filter = ttk.Combobox(
            filter_frame,
            values=['All'] + user_ids,
            state='readonly',
            width=10
        )
        self.user_filter.set('All')
        self.user_filter.pack(side='left', padx=5)
        self.user_filter.bind('<<ComboboxSelected>>', self.apply_filters)
        
        # Quick actions
        actions_frame = ttk.Frame(filter_frame)
        actions_frame.pack(side='right')
        
        self.export_btn = ttk.Button(
            actions_frame,
            text="📤 Export Results",
            command=self.export_filtered_results
        )
        self.export_btn.pack(side='left', padx=5)
        
        self.refresh_btn = ttk.Button(
            actions_frame,
            text="🔄 Refresh",
            command=self.refresh_results
        )
        self.refresh_btn.pack(side='left', padx=5)
    
    def create_results_table(self):
        """Create main results table with feedback widgets"""
        # Table frame with scrollbars
        table_frame = ttk.Frame(self)
        table_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create treeview
        columns = ('ID', 'User', 'Amount', 'Time', 'Location', 'Type', 'Risk', 'Confidence', 'Feedback')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        column_configs = {
            'ID': {'width': 80, 'anchor': 'center'},
            'User': {'width': 80, 'anchor': 'center'},
            'Amount': {'width': 100, 'anchor': 'e'},
            'Time': {'width': 80, 'anchor': 'center'},
            'Location': {'width': 100, 'anchor': 'center'},
            'Type': {'width': 100, 'anchor': 'center'},
            'Risk': {'width': 100, 'anchor': 'center'},
            'Confidence': {'width': 80, 'anchor': 'center'},
            'Feedback': {'width': 120, 'anchor': 'center'}
        }
        
        for col, config in column_configs.items():
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_column(c))
            self.tree.column(col, **config)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack table and scrollbars
        self.tree.pack(side='left', fill='both', expand=True)
        v_scrollbar.pack(side='right', fill='y')
        
        # Bind events
        self.tree.bind('<Double-1>', self.on_row_double_click)
        self.tree.bind('<Button-3>', self.on_right_click)  # Right-click context menu
        
        # Create context menu
        self.context_menu = tk.Menu(self, tearoff=0)
    
    def on_right_click(self, event):
        """Handle right-click context menu with enhanced feedback options"""
        # Select the row under cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            
            # Clear existing menu
            self.context_menu.delete(0, 'end')
            
            # Get transaction details
            transaction_id = self.tree.item(item)['values'][0]
            result = next((r for r in self.results if r['transaction_id'] == transaction_id), None)
            
            if result:
                current_risk = result.get('risk_level', 'normal')
                
                # Add menu items based on current status
                self.context_menu.add_command(label="View Details", command=self.show_transaction_details)
                self.context_menu.add_separator()
                
                # Feedback options
                if current_risk != 'normal':
                    self.context_menu.add_command(
                        label="✅ Mark as Legitimate", 
                        command=lambda: self.mark_as_legitimate(transaction_id)
                    )
                
                # Fraud marking with risk level options
                fraud_menu = tk.Menu(self.context_menu, tearoff=0)
                self.context_menu.add_cascade(label="❌ Mark as Fraud", menu=fraud_menu)
                fraud_menu.add_command(
                    label="🔴 High Risk Fraud", 
                    command=lambda: self.mark_as_fraud(transaction_id, 'high_risk')
                )
                fraud_menu.add_command(
                    label="🟠 Suspicious Fraud", 
                    command=lambda: self.mark_as_fraud(transaction_id, 'suspicious')
                )
                fraud_menu.add_command(
                    label="🟡 Low Risk Fraud", 
                    command=lambda: self.mark_as_fraud(transaction_id, 'low_risk')
                )
                
                self.context_menu.add_separator()
                self.context_menu.add_command(
                    label="🔍 Flag for Investigation", 
                    command=lambda: self.flag_for_investigation()
                )
                self.context_menu.add_separator()
                self.context_menu.add_command(label="Copy Transaction ID", command=self.copy_transaction_id)
            
            self.context_menu.post(event.x_root, event.y_root)
    
    def flag_for_investigation(self):
        """Flag transaction for manual investigation"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a transaction first.")
            return
        
        item = selection[0]
        transaction_id = self.tree.item(item)['values'][0]
        
        # Show investigation flag dialog
        self.show_investigation_dialog(transaction_id)
    
    def show_investigation_dialog(self, transaction_id):
        """Show responsive dialog for flagging transaction for investigation"""
        # Find the transaction result
        result = next((r for r in self.results if r['transaction_id'] == transaction_id), None)
        if not result:
            return
        
        # Create investigation dialog
        investigation_dialog = tk.Toplevel(self.main_window.root)
        investigation_dialog.title(f"Flag Transaction {transaction_id} for Investigation")
        investigation_dialog.geometry("600x700")
        investigation_dialog.minsize(500, 600)
        investigation_dialog.resizable(True, True)
        
        # Make dialog modal
        investigation_dialog.transient(self.main_window.root)
        investigation_dialog.grab_set()
        
        # Configure grid weights for responsiveness
        investigation_dialog.grid_rowconfigure(1, weight=1)
        investigation_dialog.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = ttk.Frame(investigation_dialog)
        header_frame.grid(row=0, column=0, sticky='ew', padx=20, pady=10)
        header_frame.grid_columnconfigure(0, weight=1)
        
        title_label = ttk.Label(
            header_frame, 
            text="🔍 Flag for Investigation", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0)
        
        # Main content frame with scrollbar
        main_frame = ttk.Frame(investigation_dialog)
        main_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid the canvas and scrollbar
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Transaction summary section
        summary_frame = ttk.LabelFrame(scrollable_frame, text="Transaction Summary", padding=15)
        summary_frame.pack(fill='x', pady=(0, 15))
        
        # Create summary in a more responsive way
        summary_data = [
            ("Transaction ID:", result['transaction_id']),
            ("User:", result['user_id']),
            ("Amount:", f"Rs. {result['amount']:,.2f}"),
            ("Time:", result['timestamp']),
            ("Location:", result['location']),
            ("Type:", result['transaction_type']),
            ("AI Assessment:", result['risk_level'].replace('_', ' ').title()),
            ("AI Confidence:", f"{result['ai_confidence']:.1%}")
        ]
        
        for i, (label, value) in enumerate(summary_data):
            row = i // 2
            col = (i % 2) * 2
            
            ttk.Label(summary_frame, text=label, font=('Arial', 9, 'bold')).grid(
                row=row, column=col, sticky='w', padx=(0, 5), pady=2
            )
            ttk.Label(summary_frame, text=value, font=('Arial', 9)).grid(
                row=row, column=col+1, sticky='w', padx=(0, 20), pady=2
            )
        
        # Configure grid weights for summary
        for i in range(4):
            summary_frame.grid_columnconfigure(i, weight=1)
        
        # Investigation reason section
        reason_frame = ttk.LabelFrame(scrollable_frame, text="Investigation Reason", padding=15)
        reason_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        ttk.Label(
            reason_frame, 
            text="Why should this transaction be investigated?", 
            font=('Arial', 11, 'bold')
        ).pack(anchor='w', pady=(0, 10))
        
        # Predefined reasons with better layout
        reason_var = tk.StringVar()
        reasons = [
            "Unusual customer behavior - needs verification",
            "Large amount requires manual approval",
            "Customer complaint or dispute",
            "Suspicious timing or location pattern",
            "AI confidence level unclear",
            "Regulatory compliance check required",
            "Custom reason (specify below)"
        ]
        
        # Create frame for radio buttons
        radio_frame = ttk.Frame(reason_frame)
        radio_frame.pack(fill='x', pady=(0, 15))
        
        for i, reason in enumerate(reasons):
            radio_btn = ttk.Radiobutton(
                radio_frame, 
                text=reason, 
                variable=reason_var, 
                value=reason
            )
            radio_btn.pack(anchor='w', pady=3)
        
        # Custom reason text with label
        custom_label = ttk.Label(reason_frame, text="Custom reason:", font=('Arial', 10, 'bold'))
        custom_label.pack(anchor='w', pady=(10, 5))
        
        custom_reason_frame = ttk.Frame(reason_frame)
        custom_reason_frame.pack(fill='both', expand=True)
        
        custom_reason_text = tk.Text(custom_reason_frame, height=4, wrap='word', font=('Arial', 9))
        custom_scrollbar = ttk.Scrollbar(custom_reason_frame, orient='vertical', command=custom_reason_text.yview)
        custom_reason_text.configure(yscrollcommand=custom_scrollbar.set)
        
        custom_reason_text.pack(side='left', fill='both', expand=True)
        custom_scrollbar.pack(side='right', fill='y')
        
        # Priority and assignment section
        priority_frame = ttk.LabelFrame(scrollable_frame, text="Priority & Assignment", padding=15)
        priority_frame.pack(fill='x', pady=(0, 15))
        
        # Priority level
        priority_row = ttk.Frame(priority_frame)
        priority_row.pack(fill='x', pady=(0, 10))
        
        ttk.Label(priority_row, text="Priority Level:", font=('Arial', 10, 'bold')).pack(side='left')
        priority_var = tk.StringVar(value="Medium")
        priority_combo = ttk.Combobox(
            priority_row, 
            textvariable=priority_var, 
            values=["Low", "Medium", "High", "Urgent"], 
            state="readonly", 
            width=15
        )
        priority_combo.pack(side='left', padx=10)
        
        # Assign to (optional)
        assign_row = ttk.Frame(priority_frame)
        assign_row.pack(fill='x')
        
        ttk.Label(assign_row, text="Assign to (optional):", font=('Arial', 10, 'bold')).pack(side='left')
        assign_var = tk.StringVar()
        assign_entry = ttk.Entry(assign_row, textvariable=assign_var, width=20)
        assign_entry.pack(side='left', padx=10)
        
        # Button frame
        button_frame = ttk.Frame(investigation_dialog)
        button_frame.grid(row=2, column=0, sticky='ew', padx=20, pady=20)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        def submit_investigation():
            selected_reason = reason_var.get()
            custom_reason = custom_reason_text.get('1.0', 'end-1c').strip()
            priority = priority_var.get()
            assigned_to = assign_var.get().strip()
            
            if not selected_reason:
                messagebox.showwarning("Missing Information", "Please select a reason for investigation.")
                return
            
            # Combine reasons
            final_reason = selected_reason
            if selected_reason == "Custom reason (specify below)" and custom_reason:
                final_reason = custom_reason
            elif custom_reason:
                final_reason += f"\nAdditional notes: {custom_reason}"
            
            # Save investigation flag
            flag_data = {
                'transaction_id': transaction_id,
                'reason': final_reason,
                'priority': priority,
                'assigned_to': assigned_to if assigned_to else None,
                'flagged_date': datetime.now().isoformat(),
                'status': 'Pending'
            }
            
            self.save_investigation_flag(flag_data)
            
            # Update display
            self.update_investigation_display(transaction_id, priority)
            
            # Update investigation tracker in main window
            if hasattr(self.main_window, 'update_investigation_tracker'):
                self.main_window.update_investigation_tracker()
            
            investigation_dialog.destroy()
            
            messagebox.showinfo(
                "Investigation Flagged", 
                f"Transaction {transaction_id} has been flagged for {priority.lower()} priority investigation."
            )
        
        def cancel_investigation():
            investigation_dialog.destroy()
        
        # Action buttons
        submit_btn = ttk.Button(
            button_frame, 
            text="🔍 Submit Investigation Flag", 
            command=submit_investigation
        )
        submit_btn.grid(row=0, column=0, sticky='ew', padx=(0, 10))
        
        cancel_btn = ttk.Button(
            button_frame, 
            text="✖ Cancel", 
            command=cancel_investigation
        )
        cancel_btn.grid(row=0, column=1, sticky='ew', padx=(10, 0))
        
        # Bind mouse wheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Focus and center the dialog
        investigation_dialog.focus()
    
    def save_investigation_flag(self, flag_data):
        """Save investigation flag to database"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.main_window.db.db_path) as conn:
                cursor = conn.cursor()
                
                # Create investigation flags table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS investigation_flags (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transaction_id TEXT,
                        reason TEXT,
                        priority TEXT,
                        assigned_to TEXT,
                        flagged_date TEXT,
                        status TEXT DEFAULT 'Pending',
                        resolution_notes TEXT,
                        resolved_date TEXT
                    )
                ''')
                
                # Insert investigation flag
                cursor.execute('''
                    INSERT INTO investigation_flags 
                    (transaction_id, reason, priority, assigned_to, flagged_date, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    flag_data['transaction_id'],
                    flag_data['reason'],
                    flag_data['priority'],
                    flag_data['assigned_to'],
                    flag_data['flagged_date'],
                    flag_data['status']
                ))
                
                conn.commit()
                
                # Add to local tracking
                self.investigation_flags.append(flag_data)
                
                logging.info(f"Investigation flag saved for transaction {flag_data['transaction_id']}")
                
        except Exception as e:
            logging.error(f"Error saving investigation flag: {e}")
    
    def load_investigation_flags(self):
        """Load existing investigation flags from database"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.main_window.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT transaction_id, reason, priority, assigned_to, flagged_date, status
                    FROM investigation_flags 
                    WHERE status = 'Pending'
                    ORDER BY flagged_date DESC
                ''')
                
                rows = cursor.fetchall()
                self.investigation_flags = []
                
                for row in rows:
                    flag_data = {
                        'transaction_id': row[0],
                        'reason': row[1],
                        'priority': row[2],
                        'assigned_to': row[3],
                        'flagged_date': row[4],
                        'status': row[5]
                    }
                    self.investigation_flags.append(flag_data)
                    
        except Exception as e:
            logging.error(f"Error loading investigation flags: {e}")
            self.investigation_flags = []
    
    def update_investigation_display(self, transaction_id, priority):
        """Update the display to show investigation flag"""
        for item in self.tree.get_children():
            if self.tree.item(item)['values'][0] == transaction_id:
                values = list(self.tree.item(item)['values'])
                values[8] = f"🔍 {priority} Priority"  # Update feedback column
                self.tree.item(item, values=values, tags=['investigation'])
                break
        
        # Configure investigation tag color
        self.configure_row_colors()
    
    def mark_as_legitimate(self, transaction_id):
        """Mark transaction as legitimate and move to normal"""
        try:
            # Find and update the transaction
            for i, result in enumerate(self.results):
                if result['transaction_id'] == transaction_id:
                    # Update the result
                    self.results[i]['risk_level'] = 'normal'
                    self.results[i]['ai_prediction'] = 1  # Normal
                    self.results[i]['ai_confidence'] = 0.9  # High confidence in legitimate
                    self.results[i]['user_feedback'] = 'legitimate'
                    self.results[i]['feedback_timestamp'] = datetime.now().isoformat()
                    break
            
            # Save feedback to system
            if hasattr(self.main_window, 'feedback_manager'):
                self.main_window.feedback_manager.add_feedback(transaction_id, 'incorrect', 'manual_legitimate')
            
            # Update display
            self.update_transaction_display(transaction_id, 'normal', '✅ Legitimate')
            
            # Refresh filtered results if needed
            self.apply_filters()
            
            self.feedback_count += 1
            self.update_display_counts()
            
            messagebox.showinfo("Feedback Saved", f"Transaction {transaction_id} marked as legitimate.")
            
        except Exception as e:
            logging.error(f"Error marking transaction as legitimate: {e}")
            messagebox.showerror("Error", f"Failed to update transaction: {str(e)}")
    
    def mark_as_fraud(self, transaction_id, risk_level):
        """Mark transaction as fraud with specified risk level"""
        try:
            # Show confirmation dialog with risk level selection
            risk_labels = {
                'high_risk': 'High Risk',
                'suspicious': 'Suspicious', 
                'low_risk': 'Low Risk'
            }
            
            confirm_msg = f"Mark transaction {transaction_id} as {risk_labels[risk_level]} fraud?"
            
            if messagebox.askyesno("Confirm Fraud Classification", confirm_msg):
                
                # Find and update the transaction
                for i, result in enumerate(self.results):
                    if result['transaction_id'] == transaction_id:
                        # Update the result
                        self.results[i]['risk_level'] = risk_level
                        self.results[i]['ai_prediction'] = -1  # Fraud
                        
                        # Set confidence based on risk level
                        confidence_map = {
                            'high_risk': 0.95,
                            'suspicious': 0.8,
                            'low_risk': 0.6
                        }
                        self.results[i]['ai_confidence'] = confidence_map[risk_level]
                        self.results[i]['user_feedback'] = f'fraud_{risk_level}'
                        self.results[i]['feedback_timestamp'] = datetime.now().isoformat()
                        break
                
                # Save feedback to system
                if hasattr(self.main_window, 'feedback_manager'):
                    self.main_window.feedback_manager.add_feedback(
                        transaction_id, 
                        'correct', 
                        f'manual_fraud_{risk_level}'
                    )
                
                # Update display
                risk_display = risk_labels[risk_level]
                self.update_transaction_display(transaction_id, risk_level, f'❌ {risk_display}')
                
                # Refresh filtered results if needed
                self.apply_filters()
                
                self.feedback_count += 1
                self.update_display_counts()
                
                messagebox.showinfo(
                    "Fraud Classification Saved", 
                    f"Transaction {transaction_id} marked as {risk_display} fraud."
                )
        
        except Exception as e:
            logging.error(f"Error marking transaction as fraud: {e}")
            messagebox.showerror("Error", f"Failed to update transaction: {str(e)}")
    
    def update_transaction_display(self, transaction_id, new_risk_level, new_feedback):
        """Update the display of a specific transaction"""
        for item in self.tree.get_children():
            if self.tree.item(item)['values'][0] == transaction_id:
                values = list(self.tree.item(item)['values'])
                
                # Update risk level display
                risk_display_map = {
                    'high_risk': 'High Risk',
                    'suspicious': 'Suspicious',
                    'low_risk': 'Low Risk',
                    'normal': 'Normal'
                }
                values[6] = risk_display_map.get(new_risk_level, new_risk_level)
                
                # Update feedback
                values[8] = new_feedback
                
                # Set appropriate tag for coloring
                self.tree.item(item, values=values, tags=[new_risk_level])
                break
        
        # Update tag colors
        self.configure_row_colors()
    
    def configure_row_colors(self):
        """Configure colors for different risk levels"""
        self.tree.tag_configure('high_risk', background='#ffebee', foreground='#c62828')
        self.tree.tag_configure('suspicious', background='#fff3e0', foreground='#f57c00')
        self.tree.tag_configure('low_risk', background='#fff8e1', foreground='#ff8f00')
        self.tree.tag_configure('normal', background='#e8f5e8', foreground='#2e7d32')
        self.tree.tag_configure('investigation', background='#fff3cd', foreground='#856404')
    
    def populate_results(self):
        """Populate results table with transaction data"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add results
        for result in self.filtered_results:
            # Determine display values
            risk_level = result['risk_level'].replace('_', ' ').title()
            confidence = "{:.1%}".format(result['ai_confidence'])
            amount = "Rs. {:.2f}".format(result['amount'])
            
            # Color coding based on risk
            tags = [result['risk_level']]
            
            # Feedback status
            feedback_status = self.get_feedback_status(result)
            
            # Insert row
            item = self.tree.insert('', 'end', values=(
                result['transaction_id'],
                result['user_id'],
                amount,
                result['timestamp'],
                result['location'],
                result['transaction_type'],
                risk_level,
                confidence,
                feedback_status
            ), tags=tags)
        
        # Configure tag colors
        self.configure_row_colors()
        
        # Update display counts
        self.update_display_counts()
    
    def get_feedback_status(self, result):
        """Get feedback status for display"""
        # Check if user provided feedback
        if result.get('user_feedback'):
            feedback_type = result['user_feedback']
            if feedback_type == 'legitimate':
                return "✅ Legitimate"
            elif feedback_type.startswith('fraud_'):
                risk_level = feedback_type.replace('fraud_', '').replace('_', ' ').title()
                return f"❌ {risk_level}"
            else:
                return "📝 Reviewed"
        
        # Check if needs feedback
        if not result.get('needs_feedback', False):
            return "Auto-skipped"
        
        return "Needs Review"
    
    def apply_filters(self, event=None):
        """Apply selected filters to results"""
        try:
            filtered = self.results.copy()
            
            # Risk level filter
            risk_filter = self.risk_filter.get()
            if risk_filter != 'All':
                risk_map = {
                    'High Risk': 'high_risk',
                    'Suspicious': 'suspicious',
                    'Low Risk': 'low_risk',
                    'Normal': 'normal'
                }
                if risk_filter in risk_map:
                    filtered = [r for r in filtered if r['risk_level'] == risk_map[risk_filter]]
            
            # User filter
            user_filter = self.user_filter.get()
            if user_filter != 'All':
                filtered = [r for r in filtered if r['user_id'] == user_filter]
            
            self.filtered_results = filtered
            self.populate_results()
            
        except Exception as e:
            logging.error(f"Error applying filters: {e}")
            # Fallback - just refresh without filters
            self.filtered_results = self.results.copy()
            self.populate_results()
    
    def sort_column(self, col):
        """Sort table by column"""
        try:
            # Get current data
            data = [(self.tree.set(item, col), item) for item in self.tree.get_children('')]
            
            # Sort data
            try:
                # Try numeric sort first (for amount column)
                if 'Rs.' in str(data[0][0]) if data else False:
                    data.sort(key=lambda x: float(x[0].replace('Rs. ', '').replace(',', '')))
                else:
                    data.sort(key=lambda x: str(x[0]))
            except (ValueError, IndexError):
                # Fall back to string sort
                data.sort(key=lambda x: str(x[0]))
            
            # Rearrange items
            for index, (_, item) in enumerate(data):
                self.tree.move(item, '', index)
                
        except Exception as e:
            logging.error(f"Error sorting column {col}: {e}")
    
    def on_row_double_click(self, event):
        """Handle double-click on table row"""
        self.show_transaction_details()
    
    def show_transaction_details(self):
        """Show detailed view of selected transaction"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a transaction to view details.")
            return
        
        item = selection[0]
        transaction_id = self.tree.item(item)['values'][0]
        
        # Find result data
        result = next((r for r in self.results if r['transaction_id'] == transaction_id), None)
        if result:
            self.show_simple_details(result)
    
    def show_simple_details(self, result):
        """Show simple transaction details"""
        detail_window = tk.Toplevel(self)
        detail_window.title(f"Transaction Details - {result['transaction_id']}")
        detail_window.geometry("600x500")
        detail_window.resizable(True, True)
        
        # Create main frame with scrollbar
        main_frame = ttk.Frame(detail_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Transaction info
        info_text = f"""
Transaction Details:

ID: {result['transaction_id']}
User: {result['user_id']}
Amount: Rs. {result['amount']:.2f}
Time: {result['timestamp']}
Location: {result['location']}
Type: {result['transaction_type']}

AI Assessment:
Risk Level: {result['risk_level'].replace('_', ' ').title()}
Confidence: {result['ai_confidence']:.1%}

Reasons for flagging:
{chr(10).join(f"• {reason}" for reason in result['explanation']['reasons']) if result['explanation']['reasons'] else "• Normal transaction pattern"}

User Feedback: {result.get('user_feedback', 'None')}
Feedback Timestamp: {result.get('feedback_timestamp', 'N/A')}

Model Version: {result.get('model_version', 'Unknown')}
        """
        
        text_widget = tk.Text(main_frame, wrap='word', font=('Courier', 10))
        text_widget.pack(fill='both', expand=True)
        text_widget.insert('1.0', info_text)
        text_widget.config(state='disabled')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=text_widget.yview)
        scrollbar.pack(side='right', fill='y')
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Close button
        button_frame = ttk.Frame(detail_window)
        button_frame.pack(fill='x', padx=10, pady=10)
        ttk.Button(button_frame, text="Close", command=detail_window.destroy).pack()
    
    def update_feedback_display(self, transaction_id, feedback):
        """Update feedback display for a transaction"""
        try:
            for item in self.tree.get_children():
                if self.tree.item(item)['values'][0] == transaction_id:
                    values = list(self.tree.item(item)['values'])
                    values[8] = "✅ Confirmed" if feedback == 'correct' else "❌ Disputed"
                    self.tree.item(item, values=values)
                    break
            
            self.feedback_count += 1
            self.update_display_counts()
        except Exception as e:
            logging.error(f"Error updating feedback display: {e}")
    
    def copy_transaction_id(self):
        """Copy selected transaction ID to clipboard"""
        try:
            selection = self.tree.selection()
            if selection:
                transaction_id = self.tree.item(selection[0])['values'][0]
                self.clipboard_clear()
                self.clipboard_append(transaction_id)
                messagebox.showinfo("Copied", f"Transaction ID {transaction_id} copied to clipboard.")
        except Exception as e:
            logging.error(f"Error copying transaction ID: {e}")
    
    def create_footer(self):
        """Create footer with feedback summary and actions"""
        footer_frame = ttk.Frame(self)
        footer_frame.pack(fill='x', padx=10, pady=10)
        
        # Feedback progress
        progress_frame = ttk.LabelFrame(footer_frame, text="Feedback Progress", padding=10)
        progress_frame.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        self.feedback_label = ttk.Label(
            progress_frame,
            text=f"Feedback provided: {self.feedback_count}/{self.flagged_transactions}",
            font=('Arial', 10)
        )
        self.feedback_label.pack(anchor='w')
        
        if self.flagged_transactions > 0:
            self.feedback_progress = ttk.Progressbar(
                progress_frame,
                length=200,
                mode='determinate',
                value=(self.feedback_count / self.flagged_transactions) * 100
            )
            self.feedback_progress.pack(fill='x', pady=5)
        
        # Quick tip
        tip_label = ttk.Label(
            progress_frame,
            text="💡 Tip: Right-click for feedback options, double-click for details",
            font=('Arial', 8),
            foreground='gray'
        )
        tip_label.pack(anchor='w', pady=2)
        
        # Action buttons
        action_frame = ttk.Frame(footer_frame)
        action_frame.pack(side='right')
        
        if hasattr(self.main_window, 'show_retraining_dialog'):
            self.retrain_btn = ttk.Button(
                action_frame,
                text="🤖 Retrain Model",
                command=self.main_window.show_retraining_dialog
            )
            self.retrain_btn.pack(side='top', pady=2)
    
    def update_display_counts(self):
        """Update feedback counts and progress"""
        try:
            displayed_count = len(self.filtered_results)
            flagged_displayed = len([r for r in self.filtered_results if r['ai_prediction'] == -1])
            
            # Update summary if needed
            status_text = f"Showing {displayed_count} of {self.total_transactions} transactions"
            if displayed_count != self.total_transactions:
                status_text += f" ({flagged_displayed} flagged)"
            
            if hasattr(self.main_window, 'update_status'):
                self.main_window.update_status(status_text)
            
            # Update feedback progress
            if hasattr(self, 'feedback_label'):
                self.feedback_label.config(text=f"Feedback provided: {self.feedback_count}/{flagged_displayed}")
                
            if hasattr(self, 'feedback_progress') and flagged_displayed > 0:
                self.feedback_progress['value'] = (self.feedback_count / flagged_displayed) * 100
                
        except Exception as e:
            logging.error(f"Error updating display counts: {e}")
    
    def export_filtered_results(self):
        """Export currently filtered results"""
        if not self.filtered_results:
            messagebox.showwarning("No Data", "No results to export.")
            return
        
        try:
            from tkinter import filedialog
            import pandas as pd
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")],
                title="Export Filtered Results"
            )
            
            if filename:
                # Prepare export data
                export_data = []
                for result in self.filtered_results:
                    export_row = {
                        'TransactionID': result['transaction_id'],
                        'UserID': result['user_id'],
                        'Amount': result['amount'],
                        'Time': result['timestamp'],
                        'Location': result['location'],
                        'Type': result['transaction_type'],
                        'RiskLevel': result['risk_level'],
                        'Confidence': result['ai_confidence'],
                        'IsFlagged': 'Yes' if result['ai_prediction'] == -1 else 'No',
                        'UserFeedback': result.get('user_feedback', 'None'),
                        'Reasons': '; '.join(result['explanation']['reasons']) if result['explanation']['reasons'] else 'Normal pattern'
                    }
                    export_data.append(export_row)
                
                df = pd.DataFrame(export_data)
                
                # Export based on file extension
                if filename.endswith('.xlsx'):
                    df.to_excel(filename, index=False)
                else:
                    df.to_csv(filename, index=False)
                
                messagebox.showinfo("Export Complete", f"Filtered results exported to:\n{filename}")
                logging.info(f"Filtered results exported: {len(export_data)} transactions")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
            logging.error(f"Export error: {e}")
    
    def refresh_results(self):
        """Refresh results display"""
        try:
            # Reload investigation flags
            self.load_investigation_flags()
            
            # Refresh display
            self.populate_results()
            
            if hasattr(self.main_window, 'update_status'):
                self.main_window.update_status("Results refreshed")
                
            # Update investigation tracker
            if hasattr(self.main_window, 'update_investigation_tracker'):
                self.main_window.update_investigation_tracker()
                
        except Exception as e:
            logging.error(f"Error refreshing results: {e}")