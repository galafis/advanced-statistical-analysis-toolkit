"""
Report Generator Module

Automated statistical report generation.

Author: Gabriel Demetrios Lafis
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime
import os


class StatisticalReportGenerator:
    """Generate automated statistical reports."""
    
    def __init__(self, title: str = "Statistical Analysis Report"):
        """
        Initialize report generator.
        
        Parameters
        ----------
        title : str
            Report title
        """
        self.title = title
        self.sections = []
        self.created_at = datetime.now()
    
    def add_section(self, title: str, content: str):
        """Add a section to the report."""
        self.sections.append({
            'title': title,
            'content': content
        })
    
    def add_descriptive_statistics(self, data: pd.DataFrame):
        """
        Add descriptive statistics section.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to analyze
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            content = "No numeric columns found in data."
        else:
            desc = data[numeric_cols].describe()
            
            content = "### Descriptive Statistics\n\n"
            content += desc.to_markdown() + "\n\n"
            
            # Add skewness and kurtosis
            content += "#### Distribution Shape\n\n"
            content += "| Variable | Skewness | Kurtosis | Interpretation |\n"
            content += "|----------|----------|----------|----------------|\n"
            
            for col in numeric_cols:
                skew = data[col].skew()
                kurt = data[col].kurtosis()
                
                if abs(skew) < 0.5:
                    skew_interp = "Fairly symmetric"
                elif skew > 0:
                    skew_interp = "Right-skewed"
                else:
                    skew_interp = "Left-skewed"
                
                if abs(kurt) < 0.5:
                    kurt_interp = "Normal-like"
                elif kurt > 0:
                    kurt_interp = "Heavy tails"
                else:
                    kurt_interp = "Light tails"
                
                content += f"| {col} | {skew:.3f} | {kurt:.3f} | {skew_interp}, {kurt_interp} |\n"
        
        self.add_section("Descriptive Statistics", content)
    
    def add_missing_values_analysis(self, data: pd.DataFrame):
        """
        Add missing values analysis section.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to analyze
        """
        missing = data.isnull().sum()
        missing_pct = (missing / len(data)) * 100
        
        if missing.sum() == 0:
            content = "✓ No missing values detected in the dataset."
        else:
            content = "### Missing Values Analysis\n\n"
            content += f"Total observations: {len(data)}\n\n"
            content += "| Column | Missing Count | Percentage |\n"
            content += "|--------|---------------|------------|\n"
            
            for col in data.columns:
                if missing[col] > 0:
                    content += f"| {col} | {missing[col]} | {missing_pct[col]:.2f}% |\n"
        
        self.add_section("Missing Values", content)
    
    def add_correlation_analysis(self, data: pd.DataFrame, method: str = 'pearson'):
        """
        Add correlation analysis section.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to analyze
        method : str
            Correlation method
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            content = "Insufficient numeric columns for correlation analysis."
        else:
            corr_matrix = data[numeric_cols].corr(method=method)
            
            content = f"### Correlation Analysis ({method.capitalize()})\n\n"
            content += corr_matrix.to_markdown() + "\n\n"
            
            # Find strong correlations
            content += "#### Strong Correlations (|r| > 0.7)\n\n"
            strong_corr = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_corr.append({
                            'var1': corr_matrix.index[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if strong_corr:
                content += "| Variable 1 | Variable 2 | Correlation |\n"
                content += "|------------|------------|-------------|\n"
                for item in strong_corr:
                    content += f"| {item['var1']} | {item['var2']} | {item['correlation']:.3f} |\n"
            else:
                content += "No strong correlations found.\n"
        
        self.add_section("Correlation Analysis", content)
    
    def add_data_quality_summary(self, data: pd.DataFrame):
        """
        Add data quality summary section.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to analyze
        """
        content = "### Data Quality Summary\n\n"
        content += f"- **Shape**: {data.shape[0]} rows × {data.shape[1]} columns\n"
        content += f"- **Duplicates**: {data.duplicated().sum()} rows\n"
        content += f"- **Missing values**: {data.isnull().sum().sum()} cells ({(data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100):.2f}%)\n"
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        content += f"- **Numeric columns**: {len(numeric_cols)}\n"
        content += f"- **Categorical columns**: {len(categorical_cols)}\n"
        
        self.add_section("Data Quality", content)
    
    def add_custom_section(self, title: str, content: str):
        """
        Add a custom section.
        
        Parameters
        ----------
        title : str
            Section title
        content : str
            Section content (Markdown format)
        """
        self.add_section(title, content)
    
    def generate_markdown(self) -> str:
        """
        Generate report in Markdown format.
        
        Returns
        -------
        report : str
            Markdown report
        """
        markdown = f"# {self.title}\n\n"
        markdown += f"**Generated**: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        markdown += "---\n\n"
        
        for section in self.sections:
            markdown += f"## {section['title']}\n\n"
            markdown += section['content'] + "\n\n"
            markdown += "---\n\n"
        
        return markdown
    
    def generate_html(self) -> str:
        """
        Generate report in HTML format.
        
        Returns
        -------
        report : str
            HTML report
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ccc;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #666;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .timestamp {{
            color: #888;
            font-style: italic;
        }}
        hr {{
            border: none;
            border-top: 1px solid #eee;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <p class="timestamp">Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr>
"""
        
        for section in self.sections:
            html += f"    <h2>{section['title']}</h2>\n"
            # Convert Markdown to HTML (basic conversion)
            content_html = section['content'].replace('\n\n', '</p><p>')
            content_html = f"<p>{content_html}</p>"
            html += f"    {content_html}\n"
            html += "    <hr>\n"
        
        html += """
</body>
</html>
"""
        return html
    
    def save_report(self, filename: str, format: str = 'markdown'):
        """
        Save report to file.
        
        Parameters
        ----------
        filename : str
            Output filename
        format : str
            Report format ('markdown' or 'html')
        """
        if format == 'markdown':
            content = self.generate_markdown()
            if not filename.endswith('.md'):
                filename += '.md'
        elif format == 'html':
            content = self.generate_html()
            if not filename.endswith('.html'):
                filename += '.html'
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Report saved to {filename}")


def generate_quick_report(
    data: pd.DataFrame,
    title: str = "Statistical Analysis Report",
    output_file: Optional[str] = None
) -> str:
    """
    Generate a quick statistical report.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to analyze
    title : str
        Report title
    output_file : str, optional
        Output filename (if None, returns string)
        
    Returns
    -------
    report : str
        Markdown report
    """
    generator = StatisticalReportGenerator(title)
    
    # Add standard sections
    generator.add_data_quality_summary(data)
    generator.add_missing_values_analysis(data)
    generator.add_descriptive_statistics(data)
    generator.add_correlation_analysis(data)
    
    if output_file:
        generator.save_report(output_file)
        return None
    else:
        return generator.generate_markdown()


def create_analysis_summary(
    test_results: Dict,
    title: str = "Hypothesis Test Results"
) -> str:
    """
    Create a summary of hypothesis test results.
    
    Parameters
    ----------
    test_results : dict
        Dictionary of test results
    title : str
        Summary title
        
    Returns
    -------
    summary : str
        Markdown summary
    """
    summary = f"## {title}\n\n"
    
    for test_name, results in test_results.items():
        summary += f"### {test_name}\n\n"
        
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, float):
                    summary += f"- **{key}**: {value:.4f}\n"
                else:
                    summary += f"- **{key}**: {value}\n"
        
        summary += "\n"
    
    return summary
