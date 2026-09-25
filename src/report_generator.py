# src/report_generator.py
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.io as pio
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import base64
import tempfile
import os

class ReportGenerator:
    """Generate PDF reports for IDS-Viz analysis"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a3e60')
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c5a7a')
        )
        
    def create_report(self, metrics, cm, feature_importance, dataset_info, model_type, is_multiclass):
        """Generate complete PDF report"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []
        
        # ========== TITLE SECTION ==========
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("IDS-Viz: Intrusion Detection System Vulnerability Report", self.title_style))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # ========== DATASET INFO ==========
        story.append(Paragraph("1. Dataset Information", self.subtitle_style))
        story.append(Spacer(1, 0.1*inch))
        
        dataset_table = [
            ["Property", "Value"],
            ["Dataset Name", dataset_info.get('name', 'Custom Upload')],
            ["Total Samples", f"{dataset_info.get('total_samples', 0):,}"],
            ["Features", str(dataset_info.get('features', 0))],
            ["Classes", str(dataset_info.get('num_classes', 0))],
        ]
        
        dataset_table_style = TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#2c5a7a')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        
        t = Table(dataset_table, colWidths=[2*inch, 2.5*inch])
        t.setStyle(dataset_table_style)
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # ========== MODEL INFO ==========
        story.append(Paragraph("2. Model Configuration", self.subtitle_style))
        story.append(Spacer(1, 0.1*inch))
        
        model_table = [
            ["Property", "Value"],
            ["Model Type", model_type],
            ["Classification Type", "Multi-Class" if is_multiclass else "Binary"],
        ]
        
        t = Table(model_table, colWidths=[2*inch, 2.5*inch])
        t.setStyle(dataset_table_style)
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # ========== PERFORMANCE METRICS ==========
        story.append(Paragraph("3. Performance Metrics", self.subtitle_style))
        story.append(Spacer(1, 0.1*inch))
        
        if is_multiclass:
            # Multi-class metrics
            metrics_table = [["Class", "Precision", "Recall", "F1-Score", "Support"]]
            
            report = metrics.get('classification_report', {})
            for class_name, values in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    if isinstance(values, dict):
                        metrics_table.append([
                            class_name[:20],
                            f"{values.get('precision', 0):.3f}",
                            f"{values.get('recall', 0):.3f}",
                            f"{values.get('f1-score', 0):.3f}",
                            f"{values.get('support', 0):.0f}"
                        ])
            
            metrics_table.append(["", "", "", "", ""])
            metrics_table.append(["Accuracy", f"{metrics.get('accuracy', 0):.3f}", "", "", ""])
            
        else:
            # Binary metrics
            metrics_table = [
                ["Metric", "Value"],
                ["Accuracy", f"{metrics.get('accuracy', 0):.3f}"],
                ["Precision", f"{metrics.get('precision', 0):.3f}"],
                ["Recall", f"{metrics.get('recall', 0):.3f}"],
                ["F1-Score", f"{metrics.get('f1_score', 0):.3f}"],
            ]
        
        t = Table(metrics_table, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        
        if is_multiclass:
            multi_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5a7a')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            t.setStyle(multi_style)
        else:
            t.setStyle(dataset_table_style)
        
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # ========== CONFUSION MATRIX ==========
        story.append(Paragraph("4. Confusion Matrix", self.subtitle_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Convert confusion matrix to text
        cm_text = ""
        if is_multiclass:
            classes = list(metrics.get('classification_report', {}).keys())
            classes = [c for c in classes if c not in ['accuracy', 'macro avg', 'weighted avg']][:5]
            
            cm_text += "         "
            for c in classes[:5]:
                cm_text += f"{c[:8]:<10}"
            cm_text += "\n"
            
            for i, row in enumerate(cm[:5]):
                cm_text += f"{classes[i][:8]:<10}"
                for val in row[:5]:
                    cm_text += f"{int(val):<10}"
                cm_text += "\n"
        else:
            cm_text = f"""
            Confusion Matrix:
            
                         Predicted
                     Benign    Attack
            Actual Benign    {int(cm[0,0])}       {int(cm[0,1])}
            Actual Attack    {int(cm[1,0])}       {int(cm[1,1])}
            """
        
        story.append(Paragraph(cm_text.replace('\n', '<br/>'), self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # ========== FEATURE IMPORTANCE ==========
        if feature_importance:
            story.append(Paragraph("5. Top Feature Importance", self.subtitle_style))
            story.append(Spacer(1, 0.1*inch))
            
            # Sort and get top 10
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            importance_table = [["Rank", "Feature", "Importance Score"]]
            for idx, (feature, importance) in enumerate(sorted_features, 1):
                importance_table.append([str(idx), feature[:30], f"{importance:.4f}"])
            
            t = Table(importance_table, colWidths=[0.7*inch, 2.5*inch, 1.5*inch])
            t.setStyle(dataset_table_style)
            story.append(t)
            story.append(Spacer(1, 0.2*inch))
        
        # ========== INTERPRETATION ==========
        story.append(Paragraph("6. Key Findings & Recommendations", self.subtitle_style))
        story.append(Spacer(1, 0.1*inch))
        
        if is_multiclass:
            report_dict = metrics.get('classification_report', {})
            worst_class = None
            worst_f1 = 1.0
            
            for class_name, values in report_dict.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(values, dict):
                    f1 = values.get('f1-score', 1)
                    if f1 < worst_f1:
                        worst_f1 = f1
                        worst_class = class_name
            
            findings = f"""
            <b>Key Findings:</b><br/>
            • Overall accuracy: {metrics.get('accuracy', 0):.1%}<br/>
            • Weakest performing class: {worst_class} (F1-Score: {worst_f1:.3f})<br/>
            • This class is where attacks are most likely to evade detection<br/><br/>
            
            <b>Recommendations:</b><br/>
            • Collect more training data for the {worst_class} class<br/>
            • Apply class balancing techniques (SMOTE or undersampling)<br/>
            • Consider ensemble methods to improve robustness
            """
        else:
            recall = metrics.get('recall', 0)
            
            findings = f"""
            <b>Key Findings:</b><br/>
            • Overall accuracy: {metrics.get('accuracy', 0):.1%}<br/>
            • Attack detection rate (Recall): {recall:.1%}<br/>
            • False Negative Rate: {(1-recall):.1%} of attacks were missed<br/><br/>
            
            <b>Recommendations:</b><br/>
            • Monitor the {int(cm[1,0]) if cm.shape == (2,2) else 0} missed attacks (False Negatives) - these are your blind spots<br/>
            • Review the {int(cm[0,1]) if cm.shape == (2,2) else 0} false alarms (False Positives)<br/>
            • Use the Feature Perturbation Simulator to identify vulnerable features
            """
        
        story.append(Paragraph(findings, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
def create_download_link(buffer, filename="ids_viz_report.pdf"):
    """Create a download button for the report"""
    b64 = base64.b64encode(buffer.getvalue()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href