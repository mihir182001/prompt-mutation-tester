import csv
import io
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm


def export_csv(run_data: dict) -> bytes:
    """
    Export a test run as a CSV file.
    
    Args:
        run_data: Dict containing test run results
        
    Returns:
        CSV content as bytes
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Write header info
    writer.writerow(['Prompt Mutation Test Report'])
    writer.writerow(['Generated', datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')])
    writer.writerow([])

    # Write prompt info
    writer.writerow(['Prompt', run_data.get('prompt', '')])
    writer.writerow(['Expected Behaviour', run_data.get('expected_behaviour', '')])
    writer.writerow(['Overall Score', f"{run_data.get('overall_score', 0)}/100"])
    writer.writerow(['Verdict', run_data.get('verdict', '')])
    writer.writerow([])

    # Write strategy results
    writer.writerow(['Strategy', 'Pass Rate', 'Avg Score', 'Failure Types', 'Status'])
    strategy_scores = run_data.get('strategy_scores', {})
    for strategy, data in strategy_scores.items():
        if isinstance(data, dict):
            pass_rate = data.get('pass_rate', 0)
            avg_score = data.get('avg_score', 0)
            failure_types = ', '.join(data.get('failure_types', [])) or '-'
            status = 'PASS' if data.get('passed') else 'FAIL'
            writer.writerow([strategy, f"{pass_rate}%", f"{avg_score}/100", failure_types, status])

    writer.writerow([])

    # Write fixed prompt if available
    if run_data.get('fixed_prompt'):
        writer.writerow(['Suggested Fix'])
        writer.writerow([run_data.get('fixed_prompt', '')])

    return output.getvalue().encode('utf-8')


def export_pdf(run_data: dict) -> bytes:
    """
    Export a test run as a professional PDF report.
    
    Args:
        run_data: Dict containing test run results
        
    Returns:
        PDF content as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    # Colors
    purple = HexColor('#7c6ff7')
    green = HexColor('#059669')
    red = HexColor('#dc2626')
    amber = HexColor('#d97706')
    light_gray = HexColor('#f9f9f9')
    dark_gray = HexColor('#333333')

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', fontSize=20, textColor=purple, spaceAfter=6, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('Subtitle', fontSize=11, textColor=HexColor('#666666'), spaceAfter=20)
    heading_style = ParagraphStyle('Heading', fontSize=13, textColor=dark_gray, spaceAfter=8, fontName='Helvetica-Bold', spaceBefore=16)
    body_style = ParagraphStyle('Body', fontSize=10, textColor=dark_gray, spaceAfter=6, leading=14)
    fix_style = ParagraphStyle('Fix', fontSize=10, textColor=HexColor('#166534'), spaceAfter=6, leading=14, backColor=HexColor('#f0fdf4'), borderPadding=8)

    story = []

    # Header
    story.append(Paragraph('🧬 Prompt Mutation Test Report', title_style))
    story.append(Paragraph(f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", subtitle_style))

    # Score banner
    score = run_data.get('overall_score', 0)
    verdict = run_data.get('verdict', '')
    score_color = green if score >= 80 else amber if score >= 55 else red

    score_data = [
        [Paragraph(f'<font size="24" color="{score_color.hexval()}"><b>{score}/100</b></font>', styles['Normal']),
         Paragraph(f'<font size="14"><b>{verdict}</b></font>', styles['Normal'])],
    ]
    score_table = Table(score_data, colWidths=[8*cm, 8*cm])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_gray),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [light_gray]),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.5*cm))

    # Prompt details
    story.append(Paragraph('Prompt Details', heading_style))
    story.append(Paragraph(f"<b>Prompt:</b> {run_data.get('prompt', '')}", body_style))
    story.append(Paragraph(f"<b>Expected behaviour:</b> {run_data.get('expected_behaviour', '')}", body_style))
    if run_data.get('test_input'):
        story.append(Paragraph(f"<b>Test input:</b> {run_data.get('test_input', '')}", body_style))

    # Mutation results table
    story.append(Paragraph('Mutation Results', heading_style))

    table_data = [['Strategy', 'Pass Rate', 'Avg Score', 'Failure Type', 'Status']]
    strategy_scores = run_data.get('strategy_scores', {})

    for strategy, data in strategy_scores.items():
        if isinstance(data, dict):
            pass_rate = data.get('pass_rate', 0)
            avg_score = data.get('avg_score', 0)
            failure_types = ', '.join(data.get('failure_types', [])) or '-'
            passed = data.get('passed', False)
            status = '✓ PASS' if passed else '✗ FAIL'
            table_data.append([strategy, f"{pass_rate}%", f"{avg_score}/100", failure_types, status])

    results_table = Table(table_data, colWidths=[4.5*cm, 2.5*cm, 2.5*cm, 4*cm, 2.5*cm])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), purple),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_gray]),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e5e5')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(results_table)

    # Fixed prompt
    if run_data.get('fixed_prompt'):
        story.append(Paragraph('Suggested Fix', heading_style))
        story.append(Paragraph(run_data.get('fixed_prompt', ''), fix_style))

    # Footer
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph('Generated by Prompt Mutation Tester v0.7.0', ParagraphStyle('Footer', fontSize=8, textColor=HexColor('#999999'))))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()