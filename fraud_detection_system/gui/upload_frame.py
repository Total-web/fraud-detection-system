import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from pathlib import Path
import logging

class UploadFrame(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.current_file = None
        self.setup_ui()
    
    def setup_ui(self):
        """Setup upload interface"""
        # Main title
        title_label = ttk.Label(
            self, 
            text="Smart Transaction Monitor",
            font=('Arial', 20, 'bold')
        )
        title_label.pack(pady=20)
        
        subtitle_label = ttk.Label(
            self,
            text="AI-Powered Fraud Detection System",
            font=('Arial', 12),
            foreground='gray'
        )
        subtitle_label.pack(pady=5)
        
        # Upload section
        upload_frame = ttk.LabelFrame(self, text="Upload Transaction Data", padding=20)
        upload_frame.pack(fill='x', padx=50, pady=30)
        
        # File selection
        file_frame = ttk.Frame(upload_frame)
        file_frame.pack(fill='x', pady=10)
        
        ttk.Label(file_frame, text="Select CSV File:").pack(anchor='w')
        
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(fill='x', pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(
            file_select_frame, 
            textvariable=self.file_path_var,
            state='readonly',
            width=60
        )
        self.file_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        self.browse_btn = ttk.Button(
            file_select_frame,
            text="📁 Browse",
            command=self.browse_file
        )
        self.browse_btn.pack(side='right')
        
        # File format info
        format_frame = ttk.Frame(upload_frame)
        format_frame.pack(fill='x', pady=10)
        
        ttk.Label(format_frame, text="Required CSV Format:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        format_info = """
Required Columns:
- TransactionID - Unique identifier for each transaction
- UserID - Customer identifier
- Amount - Transaction amount (numeric)
- Time - Transaction time (HH:MM format)
- Location - Transaction location
- Type - Transaction type (TRANSFER, WITHDRAWAL, etc.)
        """
        
        format_label = ttk.Label(
            format_frame,
            text=format_info,
            font=('Arial', 9),
            foreground='gray'
        )
        format_label.pack(anchor='w', pady=5)
        
        # Validation results
        self.validation_frame = ttk.Frame(upload_frame)
        self.validation_frame.pack(fill='x', pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(upload_frame)
        button_frame.pack(fill='x', pady=20)
        
        self.validate_btn = ttk.Button(
            button_frame,
            text="✓ Validate File",
            command=self.validate_file,
            state='disabled'
        )
        self.validate_btn.pack(side='left', padx=10)
        
        self.analyze_btn = ttk.Button(
            button_frame,
            text="🚀 Start Analysis",
            command=self.start_analysis,
            state='disabled'
        )
        self.analyze_btn.pack(side='left', padx=10)
        
        # Demo section
        demo_frame = ttk.LabelFrame(self, text="Demo Mode", padding=20)
        demo_frame.pack(fill='x', padx=50, pady=20)
        
        demo_label = ttk.Label(
            demo_frame,
            text="Don't have transaction data? Generate sample data for demonstration:",
            font=('Arial', 10)
        )
        demo_label.pack(anchor='w', pady=5)
        
        demo_button_frame = ttk.Frame(demo_frame)
        demo_button_frame.pack(fill='x', pady=10)
        
        self.demo_btn = ttk.Button(
            demo_button_frame,
            text="🎲 Generate Sample Data",
            command=self.generate_demo_data
        )
        self.demo_btn.pack(side='left')
        
        demo_info = ttk.Label(
            demo_button_frame,
            text="(Creates 200 transactions with 10% fraud patterns)",
            font=('Arial', 8),
            foreground='gray'
        )
        demo_info.pack(side='left', padx=10)
        
        # File info display
        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(fill='x', padx=50, pady=20)
    
    def browse_file(self):
        """Open file browser for CSV selection"""
        filename = filedialog.askopenfilename(
            title="Select Transaction CSV File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.current_file = filename
            self.validate_btn['state'] = 'normal'
            self.clear_validation_results()
            
            # Show basic file info
            self.show_file_info(filename)
            
            logging.info(f"Selected file: {filename}")
    
    def show_file_info(self, filename):
        """Display basic file information"""
        try:
            # Clear previous info
            for widget in self.info_frame.winfo_children():
                widget.destroy()
            
            file_path = Path(filename)
            file_size = file_path.stat().st_size
            
            # Quick peek at the file
            df = pd.read_csv(filename, nrows=5)
            
            info_text = f"""
File: {file_path.name}
Size: {file_size:,} bytes
Columns: {', '.join(df.columns.tolist())}
Sample rows: {min(len(df), 5)}
            """
            
            info_label = ttk.Label(
                self.info_frame,
                text=info_text,
                font=('Arial', 9),
                foreground='blue'
            )
            info_label.pack(anchor='w')
            
        except Exception as e:
            error_label = ttk.Label(
                self.info_frame,
                text=f"Error reading file: {str(e)}",
                foreground='red'
            )
            error_label.pack(anchor='w')
    
    def validate_file(self):
        """Validate the selected CSV file"""
        if not self.current_file:
            messagebox.showwarning("No File", "Please select a CSV file first.")
            return
        
        try:
            self.main_window.update_status("Validating file...")
            
            # Clear previous validation results
            self.clear_validation_results()
            
            # Validate using data processor
            is_valid, message, df = self.main_window.data_processor.validate_csv(self.current_file)
            
            if is_valid:
                self.show_validation_success(df, message)
                self.analyze_btn['state'] = 'normal'
                self.main_window.analyze_btn['state'] = 'normal'
                
                # Store validated data
                cleaned_data = self.main_window.data_processor.clean_data(df)
                self.main_window.current_data = cleaned_data
                self.main_window.current_filename = Path(self.current_file).name
                
                logging.info(f"File validation successful: {len(cleaned_data)} transactions")
                
            else:
                self.show_validation_error(message)
                self.analyze_btn['state'] = 'disabled'
                self.main_window.analyze_btn['state'] = 'disabled'
                
                logging.warning(f"File validation failed: {message}")
            
            self.main_window.update_status("File validation complete")
            
        except Exception as e:
            self.show_validation_error(f"Validation error: {str(e)}")
            logging.error(f"File validation error: {e}")
    
    def show_validation_success(self, df, message):
        """Display successful validation results"""
        # Success header
        success_label = ttk.Label(
            self.validation_frame,
            text="✅ Validation Successful",
            font=('Arial', 12, 'bold'),
            foreground='green'
        )
        success_label.pack(anchor='w', pady=5)
        
        # Data summary
        summary_frame = ttk.Frame(self.validation_frame)
        summary_frame.pack(fill='x', pady=5)
        
        summary_text = f"""
Total Transactions: {len(df):,}
Unique Users: {df['UserID'].nunique():,}
Date Range: {df['Time'].min()} - {df['Time'].max()}
Amount Range: Rs. {df['Amount'].min():,.2f} - Rs. {df['Amount'].max():,.2f}
Locations: {df['Location'].nunique()} unique
Transaction Types: {', '.join(df['Type'].unique())}
        """
        
        summary_label = ttk.Label(
            summary_frame,
            text=summary_text,
            font=('Arial', 9),
            foreground='darkgreen'
        )
        summary_label.pack(anchor='w')
        
        # Sample data preview
        preview_frame = ttk.LabelFrame(self.validation_frame, text="Data Preview (First 5 rows)")
        preview_frame.pack(fill='x', pady=10)
        
        # Create treeview for data preview
        columns = df.columns.tolist()
        tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=6)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        
        # Add sample data
        for _, row in df.head().iterrows():
            tree.insert('', 'end', values=row.tolist())
        
        tree.pack(fill='x', padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(preview_frame, orient='vertical', command=tree.yview)
        scrollbar.pack(side='right', fill='y')
        tree.config(yscrollcommand=scrollbar.set)
    
    def show_validation_error(self, message):
        """Display validation error"""
        error_label = ttk.Label(
            self.validation_frame,
            text="❌ Validation Failed",
            font=('Arial', 12, 'bold'),
            foreground='red'
        )
        error_label.pack(anchor='w', pady=5)
        
        error_detail = ttk.Label(
            self.validation_frame,
            text=message,
            font=('Arial', 10),
            foreground='red',
            wraplength=600
        )
        error_detail.pack(anchor='w', pady=5)
        
        # Show help for common errors
        help_frame = ttk.Frame(self.validation_frame)
        help_frame.pack(fill='x', pady=10)
        
        help_label = ttk.Label(
            help_frame,
            text="Common solutions:",
            font=('Arial', 10, 'bold')
        )
        help_label.pack(anchor='w')
        
        help_text = """
- Check that all required columns are present
- Ensure Amount column contains only numbers
- Verify Time format is HH:MM (e.g., 14:30)
- Make sure TransactionID values are unique
- Remove any empty rows or columns
        """
        
        help_detail = ttk.Label(
            help_frame,
            text=help_text,
            font=('Arial', 9),
            foreground='gray'
        )
        help_detail.pack(anchor='w')
    
    def clear_validation_results(self):
        """Clear previous validation results"""
        for widget in self.validation_frame.winfo_children():
            widget.destroy()
    
    def start_analysis(self):
        """Start the fraud detection analysis"""
        if self.main_window.current_data is None:
            messagebox.showwarning("No Data", "Please validate a CSV file first.")
            return
        
        # Start analysis in main window
        self.main_window.start_analysis()
    
    def generate_demo_data(self):
        """Generate sample data for demonstration"""
        try:
            self.main_window.update_status("Generating demo data...")
            
            # Generate sample data
            sample_data = self.main_window.data_processor.generate_sample_data(200)
            
            # Set as current data
            self.main_window.current_data = sample_data
            self.main_window.current_filename = "demo_data.csv"
            
            # Update UI
            self.file_path_var.set("Generated demo data (200 transactions)")
            
            # Show validation success for demo data
            self.clear_validation_results()
            self.show_validation_success(sample_data, "Demo data generated successfully")
            
            # Enable analysis
            self.analyze_btn['state'] = 'normal'
            self.main_window.analyze_btn['state'] = 'normal'
            
            self.main_window.update_status("Demo data ready for analysis")
            
            # Show info message
            messagebox.showinfo(
                "Demo Data Generated",
                "Sample transaction data has been generated!\n\n"
                "• 200 total transactions\n"
                "• 20 different users\n"
                "• ~10% fraudulent patterns included\n"
                "• Mix of normal and suspicious activities\n\n"
                "Click 'Start Analysis' to begin fraud detection."
            )
            
            logging.info("Demo data generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate demo data: {str(e)}")
            logging.error(f"Demo data generation error: {e}")