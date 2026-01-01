"""
Analytics Reports Module.
Generates professional HTML and JSON reports from fraud analytics results.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from src.analytics.fraud_analytics import AnalyticsResult


class AnalyticsReporter:
    """Generate analytics reports in HTML and JSON formats."""
    
    def __init__(self, results: dict[str, AnalyticsResult], name: str = "fraud_analytics"):
        self.results = results
        self.name = name
        self.generated_at = datetime.now().isoformat()
    
    def to_json(self, output_path: Optional[Path] = None) -> str:
        """Export analytics as JSON."""
        report_data = {
            "report_name": self.name,
            "generated_at": self.generated_at,
            "sections": {}
        }
        
        for key, result in self.results.items():
            report_data["sections"][key] = {
                "name": result.name,
                "summary": result.summary,
                "data": result.data,
                "actionable_items": result.actionable_items
            }
        
        json_str = json.dumps(report_data, indent=2, default=str)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(json_str)
            logger.info(f"Saved analytics JSON report to {output_path}")
        
        return json_str
    
    def to_html(self, output_path: Optional[Path] = None) -> str:
        """Export analytics as clean, professional HTML report."""
        
        # Get data
        exec_data = self.results.get("executive_summary", AnalyticsResult("", [], "")).data
        exec_data = exec_data[0] if exec_data else {}
        
        category_data = self.results.get("fraud_by_category", AnalyticsResult("", [], "")).data
        merchant_data = self.results.get("fraud_by_merchant", AnalyticsResult("", [], "")).data
        patterns_data = self.results.get("high_risk_patterns", AnalyticsResult("", [], "")).data
        anomaly_data = self.results.get("anomalies", AnalyticsResult("", [], "")).data
        anomaly_data = anomaly_data[0] if anomaly_data else {}
        time_data = self.results.get("time_trends", AnalyticsResult("", [], "")).data
        time_data = time_data[0] if time_data else {}
        
        # Chart data
        category_labels = [c.get("category", "") for c in category_data]
        category_rates = [c.get("fraud_rate_percent", 0) for c in category_data]
        category_amounts = [c.get("fraud_amount", 0) for c in category_data]
        
        hourly = time_data.get("hourly_distribution", [])
        hourly_labels = [h.get('hour', 0) for h in hourly]
        hourly_rates = [h.get("fraud_rate", 0) for h in hourly]
        
        daily = time_data.get("daily_distribution", [])
        daily_labels = [d.get("day", "") for d in daily]
        daily_rates = [d.get("fraud_rate", 0) for d in daily]
        
        # Metrics
        fraud_rate = exec_data.get("fraud_rate_percent", 0)
        fraud_count = exec_data.get("fraud_transactions", 0)
        total_count = exec_data.get("total_transactions", 0)
        fraud_amount = exec_data.get("fraud_amount", 0)
        avg_fraud = exec_data.get("avg_fraud_transaction", 0)
        avg_legit = exec_data.get("avg_legitimate_transaction", 1)
        
        # Get actionable items
        all_actions = []
        for result in self.results.values():
            all_actions.extend(result.actionable_items)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fraud Detection Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #ffffff;
            --bg-alt: #f8fafc;
            --border: #e2e8f0;
            --text: #1e293b;
            --text-muted: #64748b;
            --primary: #2563eb;
            --danger: #dc2626;
            --warning: #d97706;
            --success: #059669;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            font-size: 14px;
        }}
        
        .container {{
            max-width: 1100px;
            margin: 0 auto;
            padding: 48px 24px;
        }}
        
        /* Header */
        header {{
            border-bottom: 2px solid var(--text);
            padding-bottom: 24px;
            margin-bottom: 40px;
        }}
        
        header h1 {{
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        
        header .meta {{
            color: var(--text-muted);
            font-size: 13px;
        }}
        
        /* Sections */
        section {{
            margin-bottom: 48px;
        }}
        
        h2 {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}
        
        h3 {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        p {{
            margin-bottom: 16px;
            color: var(--text);
        }}
        
        /* Metric Grid */
        .metrics {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
            margin-bottom: 32px;
        }}
        
        .metric {{
            background: var(--bg);
            padding: 20px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 28px;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }}
        
        .metric-value.danger {{ color: var(--danger); }}
        .metric-value.warning {{ color: var(--warning); }}
        .metric-value.success {{ color: var(--success); }}
        
        .metric-label {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 24px;
        }}
        
        th {{
            text-align: left;
            padding: 10px 12px;
            background: var(--bg-alt);
            border-bottom: 2px solid var(--border);
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            color: var(--text-muted);
        }}
        
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            font-variant-numeric: tabular-nums;
        }}
        
        tr:hover td {{
            background: var(--bg-alt);
        }}
        
        .text-right {{ text-align: right; }}
        .text-danger {{ color: var(--danger); }}
        .text-warning {{ color: var(--warning); }}
        .text-success {{ color: var(--success); }}
        .font-medium {{ font-weight: 500; }}
        
        /* Inline bar */
        .bar-cell {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .inline-bar {{
            height: 6px;
            background: var(--border);
            border-radius: 3px;
            width: 80px;
            overflow: hidden;
        }}
        
        .inline-bar-fill {{
            height: 100%;
            border-radius: 3px;
        }}
        
        .inline-bar-fill.danger {{ background: var(--danger); }}
        .inline-bar-fill.warning {{ background: var(--warning); }}
        .inline-bar-fill.success {{ background: var(--success); }}
        
        /* Tag */
        .tag {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
        }}
        
        .tag-high {{ background: #fee2e2; color: #991b1b; }}
        .tag-medium {{ background: #fef3c7; color: #92400e; }}
        .tag-low {{ background: #d1fae5; color: #065f46; }}
        
        /* Charts */
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 32px;
            margin-bottom: 32px;
        }}
        
        .chart-box {{
            border: 1px solid var(--border);
            padding: 20px;
        }}
        
        .chart-box h3 {{
            margin-bottom: 16px;
        }}
        
        .chart-container {{
            height: 300px;
            position: relative;
        }}
        
        /* Summary box */
        .summary-box {{
            background: var(--bg-alt);
            border-left: 4px solid var(--primary);
            padding: 16px 20px;
            margin-bottom: 24px;
        }}
        
        .summary-box p {{
            margin: 0;
        }}
        
        /* Findings list */
        .findings {{
            list-style: none;
            padding: 0;
        }}
        
        .findings li {{
            padding: 12px 0;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: flex-start;
            gap: 12px;
        }}
        
        .findings li:last-child {{
            border-bottom: none;
        }}
        
        .finding-icon {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            flex-shrink: 0;
        }}
        
        .finding-icon.high {{ background: #fee2e2; color: #991b1b; }}
        .finding-icon.medium {{ background: #fef3c7; color: #92400e; }}
        
        /* Footer */
        footer {{
            margin-top: 48px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 12px;
        }}
        
        @media (max-width: 768px) {{
            .metrics {{ grid-template-columns: repeat(2, 1fr); }}
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Fraud Detection Analysis Report</h1>
            <p class="meta">
                Dataset: Credit Card Transactions &nbsp;|&nbsp; 
                n = {total_count:,} &nbsp;|&nbsp; 
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </p>
        </header>
        
        <!-- Executive Summary -->
        <section>
            <h2>1. Executive Summary</h2>
            <div class="summary-box">
                <p>
                    Analysis of {total_count:,} transactions identified <strong>{fraud_count:,} fraudulent cases</strong> 
                    ({fraud_rate:.2f}% fraud rate), representing <strong>${fraud_amount:,.2f}</strong> in potential losses. 
                    Mean fraudulent transaction value (${avg_fraud:.2f}) is {avg_fraud/avg_legit:.1f}x higher than 
                    legitimate transactions (${avg_legit:.2f}).
                </p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{total_count:,}</div>
                    <div class="metric-label">Total Transactions</div>
                </div>
                <div class="metric">
                    <div class="metric-value danger">{fraud_count:,}</div>
                    <div class="metric-label">Fraud Cases</div>
                </div>
                <div class="metric">
                    <div class="metric-value danger">{fraud_rate:.2f}%</div>
                    <div class="metric-label">Fraud Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value warning">${fraud_amount:,.0f}</div>
                    <div class="metric-label">Amount at Risk</div>
                </div>
            </div>
        </section>
        
        <!-- Category Analysis -->
        <section>
            <h2>2. Fraud Distribution by Category</h2>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>Fraud Rate (%)</h3>
                    <div class="chart-container">
                        <canvas id="categoryRateChart"></canvas>
                    </div>
                </div>
                <div class="chart-box">
                    <h3>Fraud Amount ($)</h3>
                    <div class="chart-container">
                        <canvas id="categoryAmountChart"></canvas>
                    </div>
                </div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th class="text-right">Transactions</th>
                        <th class="text-right">Fraud Count</th>
                        <th class="text-right">Fraud Rate</th>
                        <th class="text-right">Fraud Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''
                    <tr>
                        <td class="font-medium">{c.get("category", "N/A")}</td>
                        <td class="text-right">{c.get("total_transactions", 0):,}</td>
                        <td class="text-right">{c.get("fraud_count", 0):,}</td>
                        <td class="text-right">
                            <div class="bar-cell" style="justify-content: flex-end;">
                                <span class="{'text-danger' if c.get('fraud_rate_percent', 0) > 1 else 'text-warning' if c.get('fraud_rate_percent', 0) > 0.5 else ''}">{c.get("fraud_rate_percent", 0):.2f}%</span>
                                <div class="inline-bar">
                                    <div class="inline-bar-fill {'danger' if c.get('fraud_rate_percent', 0) > 1 else 'warning' if c.get('fraud_rate_percent', 0) > 0.5 else 'success'}" 
                                         style="width: {min(c.get('fraud_rate_percent', 0) * 50, 100)}%"></div>
                                </div>
                            </div>
                        </td>
                        <td class="text-right">${c.get("fraud_amount", 0):,.2f}</td>
                    </tr>
                    ''' for c in category_data)}
                </tbody>
            </table>
        </section>
        
        <!-- Temporal Analysis -->
        <section>
            <h2>3. Temporal Patterns</h2>
            <div class="chart-grid">
                <div class="chart-box">
                    <h3>Hourly Fraud Rate</h3>
                    <div class="chart-container">
                        <canvas id="hourlyChart"></canvas>
                    </div>
                </div>
                <div class="chart-box">
                    <h3>Daily Fraud Rate</h3>
                    <div class="chart-container">
                        <canvas id="dailyChart"></canvas>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Risk Patterns -->
        <section>
            <h2>4. Transaction Risk Patterns</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pattern</th>
                        <th>Threshold</th>
                        <th class="text-right">Sample Size</th>
                        <th class="text-right">Fraud Cases</th>
                        <th class="text-right">Fraud Rate</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''
                    <tr>
                        <td class="font-medium">{p.get("pattern", "N/A")}</td>
                        <td>{p.get("threshold", "N/A")}</td>
                        <td class="text-right">{p.get("occurrences", 0):,}</td>
                        <td class="text-right">{p.get("fraud_count", 0):,}</td>
                        <td class="text-right {'text-danger' if p.get('fraud_rate_percent', 0) > 5 else 'text-warning' if p.get('fraud_rate_percent', 0) > 1 else ''}">{p.get("fraud_rate_percent", 0):.2f}%</td>
                        <td><span class="tag tag-{p.get('risk_level', 'low').lower()}">{p.get("risk_level", "LOW")}</span></td>
                    </tr>
                    ''' for p in patterns_data)}
                </tbody>
            </table>
        </section>
        
        <!-- Anomaly Detection -->
        <section>
            <h2>5. Anomaly Detection</h2>
            <div class="summary-box">
                <p>
                    Using Z-score threshold of 3 (transactions &gt; ${anomaly_data.get('threshold_amount', 0):,.2f}), 
                    identified <strong>{anomaly_data.get('total_anomalies', 0):,} anomalous transactions</strong>. 
                    Of these, <strong>{anomaly_data.get('fraudulent_anomalies', 0):,} ({anomaly_data.get('anomaly_fraud_rate_percent', 0):.1f}%)</strong> 
                    were confirmed fraud cases.
                </p>
            </div>
            
            <div class="metrics" style="grid-template-columns: repeat(3, 1fr);">
                <div class="metric">
                    <div class="metric-value">{anomaly_data.get('total_anomalies', 0):,}</div>
                    <div class="metric-label">Anomalies Detected</div>
                </div>
                <div class="metric">
                    <div class="metric-value danger">{anomaly_data.get('fraudulent_anomalies', 0):,}</div>
                    <div class="metric-label">Confirmed Fraud</div>
                </div>
                <div class="metric">
                    <div class="metric-value warning">${anomaly_data.get('threshold_amount', 0):,.0f}</div>
                    <div class="metric-label">Alert Threshold</div>
                </div>
            </div>
        </section>
        
        <!-- High Risk Merchants -->
        <section>
            <h2>6. High-Risk Merchants</h2>
            <table>
                <thead>
                    <tr>
                        <th>Merchant</th>
                        <th class="text-right">Transactions</th>
                        <th class="text-right">Fraud Cases</th>
                        <th class="text-right">Fraud Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''
                    <tr>
                        <td class="font-medium">{m.get("merchant", "N/A")[:60]}</td>
                        <td class="text-right">{m.get("total_transactions", 0):,}</td>
                        <td class="text-right text-danger">{m.get("fraud_count", 0)}</td>
                        <td class="text-right text-danger">{m.get("fraud_rate_percent", 0):.2f}%</td>
                    </tr>
                    ''' for m in merchant_data[:10])}
                </tbody>
            </table>
        </section>
        
        <!-- Recommendations -->
        <section>
            <h2>7. Key Findings &amp; Recommendations</h2>
            <ul class="findings">
                {"".join(f'''
                <li>
                    <span class="finding-icon {'high' if any(x in action for x in ['HIGH', 'alert', 'fraud']) else 'medium'}">!</span>
                    <span>{action.replace('⚠️ ', '').replace('🎯 ', '').replace('🏪 ', '').replace('📊 ', '').replace('🚨 ', '').replace('📌 ', '').replace('🕐 ', '').replace('📅 ', '')}</span>
                </li>
                ''' for action in all_actions[:8])}
            </ul>
        </section>
        
        <footer>
            <p>Generated by Spark ETL Pipeline - Fraud Analytics Module | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </footer>
    </div>
    
    <script>
        Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
        Chart.defaults.font.size = 11;
        Chart.defaults.color = '#64748b';
        
        // Category Rate Chart
        new Chart(document.getElementById('categoryRateChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(category_labels)},
                datasets: [{{
                    data: {json.dumps(category_rates)},
                    backgroundColor: '#2563eb',
                    borderRadius: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#e2e8f0' }}, border: {{ display: false }} }},
                    y: {{ grid: {{ display: false }}, border: {{ display: false }} }}
                }}
            }}
        }});
        
        // Category Amount Chart
        new Chart(document.getElementById('categoryAmountChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(category_labels)},
                datasets: [{{
                    data: {json.dumps(category_amounts)},
                    backgroundColor: '#dc2626',
                    borderRadius: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#e2e8f0' }}, border: {{ display: false }} }},
                    y: {{ grid: {{ display: false }}, border: {{ display: false }} }}
                }}
            }}
        }});
        
        // Hourly Chart
        new Chart(document.getElementById('hourlyChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps([f"{h}:00" for h in hourly_labels])},
                datasets: [{{
                    data: {json.dumps(hourly_rates)},
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2,
                    pointBackgroundColor: '#2563eb',
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ display: false }}, border: {{ display: false }} }},
                    y: {{ grid: {{ color: '#e2e8f0' }}, border: {{ display: false }}, beginAtZero: true }}
                }}
            }}
        }});
        
        // Daily Chart
        new Chart(document.getElementById('dailyChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(daily_labels)},
                datasets: [{{
                    data: {json.dumps(daily_rates)},
                    backgroundColor: {json.dumps(['#dc2626' if r == max(daily_rates) else '#94a3b8' for r in daily_rates])},
                    borderRadius: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ display: false }}, border: {{ display: false }} }},
                    y: {{ grid: {{ color: '#e2e8f0' }}, border: {{ display: false }}, beginAtZero: true }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
            logger.info(f"Saved analytics HTML report to {output_path}")
        
        return html
