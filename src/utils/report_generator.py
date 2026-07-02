"""
Report Generator for AML Compliance Reports
"""
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import base64
from io import BytesIO

# Try importing PDF libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not installed. Install with: pip install reportlab")

class ReportGenerator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_csv_report(self, data, filename=None):
        """
        Generate CSV report from transaction data
        
        Args:
            data: DataFrame or list of dicts
            filename: Output filename (optional)
        
        Returns:
            CSV string and filename
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if filename is None:
            filename = f"aml_report_{self.timestamp}.csv"
        
        # Add metadata
        report_metadata = {
            'Report Generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Total Records': len(df),
            'Report Type': 'AML Suspicious Activity Report'
        }
        
        # Create CSV
        csv_data = df.to_csv(index=False)
        
        return csv_data, filename
    
    def generate_json_report(self, data, filename=None):
        """
        Generate JSON report for API consumption
        
        Args:
            data: Dictionary or list
            filename: Output filename (optional)
        
        Returns:
            JSON string and filename
        """
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'AML_SAR',
                'version': '1.0'
            },
            'data': data,
            'summary': self._generate_summary(data)
        }
        
        if filename is None:
            filename = f"aml_report_{self.timestamp}.json"
        
        return json.dumps(report, indent=2), filename
    
    def generate_pdf_report(self, data, title="AML Suspicious Activity Report", filename=None):
        """
        Generate PDF report for regulators
        
        Args:
            data: DataFrame or list of dicts
            title: Report title
            filename: Output filename (optional)
        
        Returns:
            PDF bytes and filename
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_html_report(data, title, filename)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if filename is None:
            filename = f"aml_report_{self.timestamp}.pdf"
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a237e'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#0d47a1'),
            spaceAfter=12
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Metadata
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"Total Records: {len(df)}", styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Summary statistics
        story.append(Paragraph("Summary Statistics", heading_style))
        summary_data = self._generate_summary(df)
        for key, value in summary_data.items():
            if key != 'data':
                story.append(Paragraph(f"• {key}: {value}", styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Data Table
        if not df.empty:
            story.append(Paragraph("Transaction Details", heading_style))
            
            # Prepare table data
            table_data = [df.columns.tolist()]
            # Limit to top 50 rows for PDF
            for _, row in df.head(50).iterrows():
                table_data.append([str(val) for val in row.values])
            
            # Create table
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            
            story.append(table)
            
            if len(df) > 50:
                story.append(Paragraph(f"... and {len(df) - 50} more records", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("This report is generated automatically by AML GNN Detection System", 
                              styles['Normal']))
        story.append(Paragraph("For regulatory compliance purposes only", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes, filename
    
    def _generate_html_report(self, data, title, filename):
        """Fallback HTML report when ReportLab not available"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #1a237e; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th {{ background-color: #1a237e; color: white; padding: 10px; text-align: left; }}
                td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .footer {{ margin-top: 40px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Records:</strong> {len(df)}</p>
            
            <div class="summary">
                <h3>Summary Statistics</h3>
                {''.join([f'<p><strong>{k}:</strong> {v}</p>' for k, v in self._generate_summary(df).items() if k != 'data'])}
            </div>
            
            <h3>Transaction Details</h3>
            <table>
                <tr>
                    {''.join([f'<th>{col}</th>' for col in df.columns.tolist()])}
                </tr>
                {''.join([f'<tr>{"".join([f"<td>{row[col]}</td>" for col in df.columns.tolist()])}</tr>' for _, row in df.head(50).iterrows()])}
            </table>
            
            {f'<p>... and {len(df) - 50} more records</p>' if len(df) > 50 else ''}
            
            <div class="footer">
                <p>This report is generated automatically by AML GNN Detection System</p>
                <p>For regulatory compliance purposes only</p>
            </div>
        </body>
        </html>
        """
        
        filename = filename or f"aml_report_{self.timestamp}.html"
        return html, filename
    
    def _generate_summary(self, data):
        """Generate summary statistics"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        summary = {
            'data': data
        }
        
        if not df.empty:
            # Risk score stats
            if 'Risk Score' in df.columns or 'risk_score' in df.columns:
                risk_col = 'Risk Score' if 'Risk Score' in df.columns else 'risk_score'
                summary['Average Risk Score'] = f"{df[risk_col].mean():.1f}%"
                summary['Max Risk Score'] = f"{df[risk_col].max():.1f}%"
                summary['High Risk (>70)'] = len(df[df[risk_col] > 70])
            
            # Pattern stats
            if 'Pattern' in df.columns or 'pattern' in df.columns:
                pattern_col = 'Pattern' if 'Pattern' in df.columns else 'pattern'
                summary['Unique Patterns'] = df[pattern_col].nunique()
                pattern_counts = df[pattern_col].value_counts().to_dict()
                for pattern, count in pattern_counts.items():
                    summary[f'  {pattern}'] = count
            
            # Amount stats
            if 'Amount' in df.columns or 'amount' in df.columns:
                amount_col = 'Amount' if 'Amount' in df.columns else 'amount'
                # Remove currency symbols
                amounts = df[amount_col].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
                amounts = pd.to_numeric(amounts, errors='coerce')
                if not amounts.empty:
                    summary['Total Amount'] = f"₹{amounts.sum():,.2f}"
                    summary['Average Amount'] = f"₹{amounts.mean():,.2f}"
        
        return summary

# Quick report functions
def generate_compliance_report(transactions_data, report_type="pdf"):
    """Quick function to generate compliance report"""
    generator = ReportGenerator()
    
    if report_type == "csv":
        return generator.generate_csv_report(transactions_data)
    elif report_type == "json":
        return generator.generate_json_report(transactions_data)
    elif report_type == "pdf":
        return generator.generate_pdf_report(transactions_data, "AML Compliance Report")
    else:
        return generator.generate_pdf_report(transactions_data, "AML Compliance Report")

# Export for dashboard
def get_report_download_links(data, report_name="AML_Report"):
    """Generate download links for all report formats"""
    generator = ReportGenerator()
    
    links = {}
    
    # CSV
    csv_data, csv_filename = generator.generate_csv_report(data, f"{report_name}.csv")
    links['csv'] = {
        'data': csv_data,
        'filename': csv_filename,
        'mime': 'text/csv'
    }
    
    # JSON
    json_data, json_filename = generator.generate_json_report(data, f"{report_name}.json")
    links['json'] = {
        'data': json_data,
        'filename': json_filename,
        'mime': 'application/json'
    }
    
    # PDF
    pdf_data, pdf_filename = generator.generate_pdf_report(data, f"{report_name}", f"{report_name}.pdf")
    links['pdf'] = {
        'data': pdf_data,
        'filename': pdf_filename,
        'mime': 'application/pdf'
    }
    
    return links