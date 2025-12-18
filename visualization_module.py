# ============================================================================
# VISUALIZATION FUNCTIONS - UPDATED FOR BALANCED SAMPLING
# ============================================================================


def create_timeline_visualization(results_df, output_dir):
    """Enhanced timeline with line plot + confidence bands"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    pdf = results_df.select("timestamp", "inference_time_ms", "gpu_used").toPandas()
    if len(pdf) == 0:
        return
    
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
    pdf = pdf.sort_values('timestamp')
    start_time = pdf['timestamp'].min()
    pdf['elapsed_sec'] = (pdf['timestamp'] - start_time).dt.total_seconds()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use different colors for different GPUs if available
    if pdf['gpu_used'].nunique() > 1:
        for gpu in pdf['gpu_used'].unique():
            gpu_data = pdf[pdf['gpu_used'] == gpu]
            ax.scatter(gpu_data['elapsed_sec'], gpu_data['inference_time_ms'], 
                      alpha=0.6, s=80, label=gpu, edgecolors='black', linewidth=0.5)
        ax.legend(loc='upper right', framealpha=0.9)
    else:
        # Single GPU - use gradient coloring by sequence
        scatter = ax.scatter(pdf['elapsed_sec'], pdf['inference_time_ms'], 
                           c=range(len(pdf)), cmap='viridis', 
                           alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        if len(pdf) > 1:
            z = np.polyfit(pdf['elapsed_sec'], pdf['inference_time_ms'], 1)
            p = np.poly1d(z)
            ax.plot(pdf['elapsed_sec'], p(pdf['elapsed_sec']), 
                   "--", color='red', linewidth=2, alpha=0.7, 
                   label=f'Trend: {z[0]:.1f} ms/image')
            ax.legend()
    
    # Add horizontal mean line
    mean_latency = pdf['inference_time_ms'].mean()
    ax.axhline(mean_latency, color='#e74c3c', linestyle='--', 
              linewidth=2, label=f'Mean: {mean_latency:.1f}ms', alpha=0.8)
    
    # Styling
    ax.set_xlabel('Time Elapsed (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inference Time (ms)', fontsize=13, fontweight='bold')
    ax.set_title('Processing Timeline - Latency Over Time', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add statistics box
    stats_text = f'Min: {pdf["inference_time_ms"].min():.0f}ms\n'
    stats_text += f'Max: {pdf["inference_time_ms"].max():.0f}ms\n'
    stats_text += f'Std: {pdf["inference_time_ms"].std():.0f}ms'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'processing_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Timeline saved")


def create_comprehensive_dashboard(results_df, analytics, output_dir):
    """All-in-one dashboard with latency, throughput, and performance metrics"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Latency percentiles
    ax1 = fig.add_subplot(gs[0, 0])
    lat = analytics['latency_statistics']
    if len(lat) > 0 and lat['mean_ms'].notna().any():
        labels = ['Mean', 'P50', 'P95', 'P99']
        values = [lat['mean_ms'].values[0], lat['p50_ms'].values[0],
                 lat['p95_ms'].values[0], lat['p99_ms'].values[0]]
        colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
        bars = ax1.bar(labels, values, color=colors, edgecolor='black')
        ax1.set_ylabel('Milliseconds', fontweight='bold')
        ax1.set_title('Latency Percentiles', fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom')
    
    # 2. Throughput
    ax2 = fig.add_subplot(gs[0, 1])
    tp = analytics['throughput_metrics']
    if len(tp) > 0:
        throughput = tp['throughput_img_per_sec'].values[0]
        ax2.text(0.5, 0.5, f'{throughput:.2f}\nimages/sec',
                ha='center', va='center', fontsize=28, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Throughput', fontweight='bold')
    
    # 3. Success rate
    ax3 = fig.add_subplot(gs[0, 2])
    err = analytics['error_statistics']
    if len(err) > 0:
        success_row = err[err['status'] == 'Success']
        if len(success_row) > 0:
            success_pct = success_row['percentage'].values[0]
            ax3.pie([success_pct, 100-success_pct],
                   colors=['#2ecc71', '#e74c3c'],
                   startangle=90, autopct='%1.1f%%')
            ax3.set_title('Success Rate', fontweight='bold')
    
    # 4. ENHANCED Processing Timeline
    ax5 = fig.add_subplot(gs[1, :])
    pdf = results_df.select("timestamp", "inference_time_ms").toPandas()
    pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
    pdf = pdf.sort_values('timestamp')
    
    if len(pdf) > 0:
        start_time = pdf['timestamp'].min()
        pdf['elapsed_sec'] = (pdf['timestamp'] - start_time).dt.total_seconds()
        
        # ===== FIX: Better labeling for sequential processing =====
        time_range = pdf['elapsed_sec'].max() - pdf['elapsed_sec'].min()
        
        if time_range < 1.0:  # Less than 1 second total - use sequential numbering
            pdf['x_axis'] = range(1, len(pdf) + 1)  # Start from 1, not 0
            x_label = 'Image Number'  # ← CLEARER LABEL
            x_data = pdf['x_axis']
        else:  # Actual time difference exists
            x_label = 'Time Elapsed (sec)'
            x_data = pdf['elapsed_sec']
        # ================================================================
        
        # Line plot with markers
        ax5.plot(x_data, pdf['inference_time_ms'], 
                marker='o', markersize=8, linewidth=2.5, 
                color='#3498db', alpha=0.8, label='Latency')
        
        # Add mean line
        mean_lat = pdf['inference_time_ms'].mean()
        ax5.axhline(mean_lat, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {mean_lat:.1f}ms', alpha=0.7)
        
        # Fill area for visual appeal
        ax5.fill_between(x_data, pdf['inference_time_ms'], 
                        alpha=0.2, color='#3498db')
        
        ax5.set_xlabel(x_label, fontweight='bold', fontsize=12)
        ax5.set_ylabel('Latency (ms)', fontweight='bold', fontsize=12)
        ax5.set_title('Processing Timeline', fontweight='bold', fontsize=14)
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=11)
    
    # 5. Summary
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    summary_text = [
        f"Mean Latency: {lat['mean_ms'].values[0]:.1f} ms",
        f"P95 Latency: {lat['p95_ms'].values[0]:.1f} ms",
        f"Throughput: {throughput:.2f} img/s",
        f"Total Images: {int(tp['total_images'].values[0])}"
    ]
    ax6.text(0.5, 0.5, '\n'.join(summary_text),
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('MedGemma Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(Path(output_dir) / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Dashboard saved")


def create_gpu_utilization_chart(results_df, output_dir):
    """GPU workload distribution pie + bar charts"""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    pdf = results_df.select("gpu_used").toPandas()
    if len(pdf) == 0:
        return
    
    gpu_counts = pdf['gpu_used'].value_counts()
    if len(gpu_counts) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    
    # Pie chart
    ax1.pie(gpu_counts.values, labels=gpu_counts.index, autopct='%1.1f%%',
           colors=colors[:len(gpu_counts)], startangle=90)
    ax1.set_title('GPU Workload Distribution (%)', fontsize=14, fontweight='bold')
    
    # Bar chart
    ax2.bar(range(len(gpu_counts)), gpu_counts.values, color=colors[:len(gpu_counts)],
           edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('GPU', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Images', fontsize=13, fontweight='bold')
    ax2.set_title('GPU Workload Distribution (Count)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(gpu_counts)))
    ax2.set_xticklabels([g.split('-')[0] if '-' in g else g for g in gpu_counts.index],
                        rotation=45, ha='right')
    
    for i, v in enumerate(gpu_counts.values):
        ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'gpu_utilization.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ GPU utilization saved")


def create_scalability_comparison(summary_csv_path, output_dir):
    """Compare throughput/latency across different load sizes"""
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    
    if not Path(summary_csv_path).exists():
        print("  ⚠️  Summary CSV not found")
        return
    
    df = pd.read_csv(summary_csv_path)
    if len(df) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    
    # Throughput (CHANGED TO images/sec)
    ax1.bar(df['num_images'], df['wall_throughput_img_per_sec'],
            color=colors[:len(df)], edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Number of Images', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Throughput (images/sec)', fontsize=13, fontweight='bold')  # ← CHANGED
    ax1.set_title('Scalability: Throughput vs Load Size', fontsize=15, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for i, row in df.iterrows():
        if pd.notna(row['wall_throughput_img_per_sec']):
            ax1.text(row['num_images'], row['wall_throughput_img_per_sec'],
                    f"{row['wall_throughput_img_per_sec']:.2f}",
                    ha='center', va='bottom', fontweight='bold')
    
    # Latency
    if 'mean_latency_ms' in df.columns:
        ax2.bar(df['num_images'], df['mean_latency_ms'],
                color=colors[:len(df)], edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Number of Images', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Mean Latency (ms)', fontsize=13, fontweight='bold')
        ax2.set_title('Scalability: Latency vs Load Size', fontsize=15, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for i, row in df.iterrows():
            if pd.notna(row['mean_latency_ms']):
                ax2.text(row['num_images'], row['mean_latency_ms'],
                        f"{row['mean_latency_ms']:.0f}",
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'scalability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Scalability comparison saved")


def create_xray_report_visualization(results_df, output_dir, num_samples=10):
    """HTML gallery showing X-rays side-by-side with clinical reports
    
    FIXED: Now properly handles balanced DataFrame input (5 NORMAL + 5 PNEUMONIA)
    """
    from PIL import Image
    import base64
    from io import BytesIO
    from pathlib import Path
    import pandas as pd
    import re
    from datetime import datetime

    print(f"  Creating X-ray + Report gallery ({num_samples} samples)...")

    # ===== FIX: Convert to pandas first, then limit =====
    # This preserves the balance already created by the pipeline
    pdf = results_df.toPandas()
    
    if len(pdf) == 0:
        print("  ⚠️  No data to visualize")
        return None
    
    # Use only the samples we have (already balanced if passed correctly)
    pdf = pdf.head(num_samples)
    # ===================================================

    html_template = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Radiology Reports</title>
<style>
* { box-sizing: border-box; }
body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f7fa;padding:20px;margin:0}
.container{max-width:1600px;margin:0 auto}
h1{text-align:center;color:#2c3e50;font-size:32px;margin-bottom:5px}
.subtitle{text-align:center;color:#7f8c8d;font-size:14px;margin-bottom:30px}

/* Card Layout */
.sample-card{
  background:white;
  border-radius:8px;
  margin-bottom:25px;
  box-shadow:0 2px 8px rgba(0,0,0,0.1);
  overflow:hidden;
  display:flex;
  flex-direction:row;
}

/* X-ray Section */
.xray-section{
  flex:0 0 450px;
  background:#2c3e50;
  padding:20px;
  display:flex;
  flex-direction:column;
  justify-content:center;
  align-items:center;
}

.xray-img{
  width:100%;
  max-width:400px;
  height:auto;
  border-radius:5px;
  border:3px solid #34495e;
  display:block;
  margin:0 auto;
}

.case-number{
  color:white;
  font-size:14px;
  font-weight:bold;
  margin-bottom:15px;
  background:#e74c3c;
  padding:8px 20px;
  border-radius:20px;
  text-align:center;
}

/* Report Section */
.report-section{
  flex:1;
  padding:30px;
  background:white;
  min-width:0;
}

.report-header{
  border-bottom:3px solid #3498db;
  padding-bottom:10px;
  margin-bottom:20px;
}

.report-title{
  font-size:20px;
  font-weight:bold;
  color:#2c3e50;
  margin:0;
}

.patient-info{
  display:grid;
  grid-template-columns:repeat(2,1fr);
  gap:8px;
  margin:15px 0;
  padding:15px;
  background:#ecf0f1;
  border-radius:5px;
  font-size:13px;
}

.info-item{display:flex}
.info-label{font-weight:bold;color:#34495e;min-width:100px}
.info-value{color:#2c3e50}

/* Report Content */
.report-content{
  font-family:'Segoe UI',Tahoma,sans-serif;
  font-size:14px;
  line-height:1.8;
  color:#2c3e50;
  margin-bottom:20px;
}

.report-content h3{
  color:#2980b9;
  font-size:16px;
  margin-top:20px;
  margin-bottom:10px;
  border-left:4px solid #3498db;
  padding-left:10px;
}

.report-content ul{margin:10px 0;padding-left:25px}
.report-content li{margin:5px 0}
.report-content p{margin:10px 0}

.section-header{
  font-weight:bold;
  color:#2980b9;
  margin-top:15px;
  margin-bottom:8px;
  font-size:15px;
}

/* Performance Metrics */
.performance-section{
  margin-top:20px;
  padding-top:15px;
  border-top:2px solid #ecf0f1;
}

.performance-title{
  font-size:11px;
  color:#95a5a6;
  text-transform:uppercase;
  letter-spacing:0.5px;
  margin-bottom:10px;
  font-weight:600;
}

.metrics-grid{
  display:grid;
  grid-template-columns:repeat(3,1fr);
  gap:15px;
}

.metric-card{
  background:#f8f9fa;
  padding:12px;
  border-radius:5px;
  text-align:center;
  border-left:3px solid #3498db;
}

.metric-card.fast{border-left-color:#27ae60}
.metric-card.medium{border-left-color:#f39c12}
.metric-card.slow{border-left-color:#e74c3c}

.metric-label{
  font-size:10px;
  color:#7f8c8d;
  text-transform:uppercase;
  margin-bottom:5px;
  letter-spacing:0.3px;
}

.metric-value{
  font-size:18px;
  font-weight:bold;
  color:#2c3e50;
}

.metric-unit{
  font-size:11px;
  color:#95a5a6;
  margin-left:2px;
}

.metric-subtitle{
  font-size:9px;
  color:#95a5a6;
  margin-top:3px;
}

/* ========================================
   RESPONSIVE DESIGN
   ======================================== */

/* Tablets (landscape) */
@media (max-width: 1200px) {
  .container { max-width: 100%; padding: 0 15px; }
  .xray-section { flex: 0 0 400px; }
  .report-section { padding: 25px; }
}

/* Tablets (portrait) */
@media (max-width: 992px) {
  h1 { font-size: 28px; }
  .subtitle { font-size: 13px; }

  .sample-card {
    flex-direction: column;
  }

  .xray-section {
    flex: 1 1 auto;
    width: 100%;
    padding: 30px 20px;
  }

  .xray-img {
    max-width: 350px;
  }

  .patient-info {
    grid-template-columns: 1fr;
  }
}

/* Mobile (large) */
@media (max-width: 768px) {
  body { padding: 10px; }
  h1 { font-size: 24px; }
  .subtitle { font-size: 12px; margin-bottom: 20px; }

  .report-section { padding: 20px; }
  .report-title { font-size: 18px; }

  .xray-img {
    max-width: 100%;
  }

  .metrics-grid {
    grid-template-columns: 1fr;
    gap: 10px;
  }

  .metric-value { font-size: 16px; }
}

/* Mobile (small) */
@media (max-width: 480px) {
  body { padding: 5px; }
  h1 { font-size: 20px; }
  .sample-card { margin-bottom: 15px; }
  .report-section { padding: 15px; }
  .patient-info { padding: 12px; font-size: 12px; }
  .report-content { font-size: 13px; }
  .info-label { min-width: 80px; }
}

/* Print Styles */
@media print {
  body { background: white; }
  .sample-card { page-break-after: always; }
  .performance-section { display: none; }
}
</style>
</head><body>
<div class="container">
<h1>📋 Radiology Report Archive</h1>
<div class="subtitle">AI-Powered Chest X-Ray Diagnostic Reports · Real-Time Clinical Decision Support</div>
{samples_html}
</div></body></html>"""

    def clean_report(raw_report):
        """Clean up AI-generated report - extract only clinical content"""
        if not raw_report or raw_report == "Error" or pd.isna(raw_report):
            return "<p><strong>Report generation failed.</strong> Please review image manually.</p>"
        
        report = str(raw_report)
        
        # ===== REMOVE AI PREAMBLES (aggressive) =====
        report = re.sub(r'^(Okay,?\s*)?(I\s+will\s+analyze|I\s+have\s+analyzed|Here\s+is|Let\s+me\s+provide|Based\s+on).*?report[:\.]?\s*', 
                       '', report, flags=re.IGNORECASE | re.MULTILINE)
        report = re.sub(r'^\s*Okay[,\.]?\s*', '', report, flags=re.IGNORECASE)
        
        # Remove duplicate header sections
        report = re.sub(r'\*\*Radiology Report\*\*.*?(?=\*\*I\.|I\.|\*\*II\.|II\.)', '', 
                       report, flags=re.DOTALL)
        
        # Remove duplicate metadata
        for pattern in [r'\*\*Patient:\*\*.*?\n', r'\*\*Date:\*\*.*?\n', 
                       r'\*\*Study Type:\*\*.*?\n', r'\*\*Quality:\*\*.*?\n',
                       r'\*\*Radiologist:\*\*.*?\n', r'Radiologist:.*?\n',
                       r'\*\*Signature:\*\*.*?\n', r'Signature:.*?\n']:
            report = re.sub(pattern, '', report, flags=re.IGNORECASE)
        
        # Convert markdown bold to HTML
        report = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', report)
        
        # Process line by line
        lines = report.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                continue
            
            # Skip unwanted lines
            if any(kw in stripped.lower() for kw in ['radiologist:', 'signature:', 
                                                       'okay,', 'i will analyze', 
                                                       'here is', 'based on']):
                continue
            
            # ===== SECTION HEADERS (I., II., III., IV., V., VI.) =====
            if re.match(r'^(I{1,3}|IV|V|VI)\.?\s+', stripped):
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                formatted_lines.append(f'<h3>{stripped}</h3>')
            
            # ===== SUB-HEADERS (remove asterisks completely) =====
            elif re.match(r'^\*?\s*<strong>[^:]+:</strong>', stripped):
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                # Remove leading asterisk and space
                cleaned = re.sub(r'^\*\s*', '', stripped)
                formatted_lines.append(f'<div class="section-header">{cleaned}</div>')
            
            # ===== BULLET POINTS =====
            elif stripped.startswith('*') or stripped.startswith('-') or stripped.startswith('•'):
                item_text = re.sub(r'^[\*\-•]\s*', '', stripped)
                if item_text:
                    if not in_list:
                        formatted_lines.append('<ul>')
                        in_list = True
                    formatted_lines.append(f'<li>{item_text}</li>')
            
            # ===== REGULAR TEXT =====
            else:
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                formatted_lines.append(f'<p>{stripped}</p>')
        
        if in_list:
            formatted_lines.append('</ul>')
        
        html_report = '\n'.join(formatted_lines)
        return html_report.strip() if html_report.strip() else "<p>No report content available.</p>"

    samples_html = []
    for idx, row in pdf.iterrows():
        try:
            # Load and encode image
            img = Image.open(row['file_path']).convert('RGB')
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Extract data
            raw_report = row['report_text'] if pd.notna(row['report_text']) else "Error"
            cleaned_report = clean_report(raw_report)

            # Performance metrics
            latency_ms = row['inference_time_ms'] if pd.notna(row['inference_time_ms']) else 0
            latency_sec = latency_ms / 1000.0
            gpu = row['gpu_used'] if pd.notna(row['gpu_used']) else "Unknown"
            gpu_short = gpu.split('-')[0] if '-' in gpu else gpu

            # Latency class for color coding
            if latency_ms < 5000:
                latency_class = "fast"
            elif latency_ms < 15000:
                latency_class = "medium"
            else:
                latency_class = "slow"

            # Throughput
            throughput_per_sec = 1.0 / latency_sec if latency_sec > 0 else 0

            # ===== MEANINGFUL METADATA (Clean & Professional) =====
            filepath = Path(row['file_path'])
            filename = filepath.stem
            
            # Extract Study ID from IM-0545 format
            study_match = re.search(r'IM-(\d+)', filename, re.IGNORECASE)
            if study_match:
                study_id = f"IM-{study_match.group(1)}"
            else:
                study_id = f"STUDY-{abs(hash(filename)) % 100000:05d}"
            
            # Case ID
            case_id = f"CXR-TRAIN-{study_id}"

            # Report date (no confusing timestamps)
            if pd.notna(row['timestamp']):
                report_date = pd.to_datetime(row['timestamp']).strftime("%B %d, %Y")
            else:
                report_date = datetime.now().strftime("%B %d, %Y")
            
            # Modality
            modality = "Chest X-Ray (PA)"

            sample_html = f"""
<div class="sample-card">
  <div class="xray-section">
    <div class="case-number">Case #{idx + 1}</div>
    <img class="xray-img" src="data:image/png;base64,{img_base64}" alt="Chest X-ray">
  </div>
  <div class="report-section">
    <div class="report-header">
      <div class="report-title">RADIOLOGY REPORT - Chest X-Ray</div>
      <div class="patient-info">
        <div class="info-item"><span class="info-label">Study ID:</span><span class="info-value">{study_id}</span></div>
        <div class="info-item"><span class="info-label">Case ID:</span><span class="info-value">{case_id}</span></div>
        <div class="info-item"><span class="info-label">Modality:</span><span class="info-value">{modality}</span></div>
        <div class="info-item"><span class="info-label">Report Date:</span><span class="info-value">{report_date}</span></div>
      </div>
    </div>
    <div class="report-content">
      {cleaned_report}
    </div>
    <div class="performance-section">
      <div class="performance-title">Performance Metrics</div>
      <div class="metrics-grid">
        <div class="metric-card {latency_class}">
          <div class="metric-label">Processing Time</div>
          <div class="metric-value">{latency_sec:.2f}<span class="metric-unit">sec</span></div>
          <div class="metric-subtitle">{latency_ms:.0f} ms</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Throughput</div>
          <div class="metric-value">{throughput_per_sec:.2f}<span class="metric-unit">/sec</span></div>
          <div class="metric-subtitle">images per second</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Compute Unit</div>
          <div class="metric-value">{gpu_short}</div>
          <div class="metric-subtitle">GPU accelerated</div>
        </div>
      </div>
    </div>
  </div>
</div>"""
            samples_html.append(sample_html)
        except Exception as e:
            print(f"    ⚠️  Error processing sample {idx+1}: {e}")
            continue

    final_html = html_template.replace('{samples_html}', '\n'.join(samples_html))
    final_html = final_html.replace('{num_samples}', str(len(samples_html)))

    output_path = Path(output_dir) / "xray_reports_visualization.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)

    print(f"  ✅ HTML gallery saved: {output_path}")
    return output_path