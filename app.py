from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import sys
import os

# Add prompt_mutator folder to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'prompt_mutator'))

from mutator import generate_mutations
from runner import run_all_mutations
from judge import judge_all_results
from reporter import aggregate_scores
from fixer import suggest_fix
from database import db, save_test_run, get_all_runs, get_run_by_id

# Create the Flask app
app = Flask(__name__)

# Configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prompt_mutator.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Connect database to app
db.init_app(app)

# Create tables on startup
with app.app_context():
    db.create_all()


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/run', methods=['POST'])
def run_test():
    """Run mutation tests and return results as JSON."""
    data = request.json
    prompt = data.get('prompt', '')
    expected = data.get('expected', '')
    test_input = data.get('test_input', '')
    strategies = data.get('strategies', None)

    if not prompt or not expected:
        return jsonify({'error': 'Prompt and expected behaviour are required'}), 400

    try:
        mutations = generate_mutations(prompt, strategies)
        results = run_all_mutations(mutations, [test_input])
        judged = judge_all_results(prompt, expected, results)
        report = aggregate_scores(judged)

        fixed_prompt = None
        if report['overall_score'] < 80:
            fixed_prompt = suggest_fix(prompt, expected, judged)

        strategy_scores = {
            strategy: {
                'pass_rate': round(s['pass_rate'] * 100),
                'avg_score': round(s['avg_score'] * 100),
                'failure_types': s['failure_types'],
                'passed': s['pass_rate'] == 1.0,
            }
            for strategy, s in report['strategy_scores'].items()
        }

        run = save_test_run(
            prompt=prompt,
            expected_behaviour=expected,
            test_input=test_input,
            overall_score=report['overall_score'],
            strategy_scores=strategy_scores,
            fixed_prompt=fixed_prompt,
        )

        return jsonify({
            'run_id': run.id,
            'overall_score': report['overall_score'],
            'strategy_scores': strategy_scores,
            'fixed_prompt': fixed_prompt,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Return all past test runs."""
    runs = get_all_runs()
    return jsonify(runs)


@app.route('/run/<int:run_id>')
def get_run(run_id):
    """Return a single test run by ID."""
    run = get_run_by_id(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404
    return jsonify(run)


@app.route('/batch', methods=['POST'])
def batch_test():
    """Run mutation tests on multiple prompts at once."""
    data = request.json
    prompts = data.get('prompts', [])

    if not prompts:
        return jsonify({'error': 'No prompts provided'}), 400

    batch_results = []

    for item in prompts:
        prompt = item.get('prompt', '')
        expected = item.get('expected', '')
        test_input = item.get('test_input', '')

        if not prompt or not expected:
            continue

        try:
            mutations = generate_mutations(prompt)
            results = run_all_mutations(mutations, [test_input])
            judged = judge_all_results(prompt, expected, results)
            report = aggregate_scores(judged)

            fixed_prompt = None
            if report['overall_score'] < 80:
                fixed_prompt = suggest_fix(prompt, expected, judged)

            strategy_scores = {
                strategy: {
                    'pass_rate': round(s['pass_rate'] * 100),
                    'avg_score': round(s['avg_score'] * 100),
                    'failure_types': s['failure_types'],
                    'passed': s['pass_rate'] == 1.0,
                }
                for strategy, s in report['strategy_scores'].items()
            }

            run = save_test_run(
                prompt=prompt,
                expected_behaviour=expected,
                test_input=test_input,
                overall_score=report['overall_score'],
                strategy_scores=strategy_scores,
                fixed_prompt=fixed_prompt,
            )

            batch_results.append({
                'run_id': run.id,
                'prompt': prompt[:60] + '...' if len(prompt) > 60 else prompt,
                'overall_score': report['overall_score'],
                'fixed_prompt': fixed_prompt,
            })

        except Exception as e:
            batch_results.append({
                'prompt': prompt[:60],
                'error': str(e),
            })

    return jsonify(batch_results)


@app.route('/compare', methods=['POST'])
def compare_models():
    """Run mutation tests on the same prompt across multiple models."""
    data = request.json
    prompt = data.get('prompt', '')
    expected = data.get('expected', '')
    test_input = data.get('test_input', '')
    models = data.get('models', ['llama-3.3-70b-versatile'])

    if not prompt or not expected:
        return jsonify({'error': 'Prompt and expected behaviour are required'}), 400

    compare_results = []

    for model in models:
        try:
            mutations = generate_mutations(prompt)
            results = run_all_mutations(mutations, [test_input], model)
            judged = judge_all_results(prompt, expected, results)
            report = aggregate_scores(judged)

            strategy_scores = {
                strategy: {
                    'pass_rate': round(s['pass_rate'] * 100),
                    'avg_score': round(s['avg_score'] * 100),
                    'failure_types': s['failure_types'],
                    'passed': s['pass_rate'] == 1.0,
                }
                for strategy, s in report['strategy_scores'].items()
            }

            score = report['overall_score']
            verdict = 'ROBUST' if score >= 80 else 'MODERATE' if score >= 55 else 'FRAGILE'

            compare_results.append({
                'model': model,
                'overall_score': score,
                'verdict': verdict,
                'strategy_scores': strategy_scores,
            })

        except Exception as e:
            compare_results.append({
                'model': model,
                'error': str(e),
            })

    return jsonify(compare_results)


@app.route('/optimize', methods=['POST'])
def optimize():
    """Iteratively optimize a prompt until it scores above target."""
    data = request.json
    prompt = data.get('prompt', '')
    expected = data.get('expected', '')
    test_input = data.get('test_input', '')
    target_score = data.get('target_score', 90.0)
    max_iterations = data.get('max_iterations', 5)

    if not prompt or not expected:
        return jsonify({'error': 'Prompt and expected behaviour are required'}), 400

    try:
        from optimizer import iterative_optimize
        result = iterative_optimize(
            prompt=prompt,
            expected_behaviour=expected,
            test_inputs=[test_input],
            target_score=target_score,
            max_iterations=max_iterations,
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/traces')
def get_traces():
    """Return all LLM call traces."""
    from llm_tracer import get_traces, get_trace_stats
    return jsonify({
        'traces': get_traces(),
        'stats': get_trace_stats(),
    })


@app.route('/traces/clear', methods=['POST'])
def clear_traces_route():
    """Clear all traces."""
    from llm_tracer import clear_traces
    clear_traces()
    return jsonify({'message': 'Traces cleared'})


@app.route('/rag', methods=['POST'])
def rag_test():
    """Test a RAG extraction prompt against an uploaded document."""
    from rag_tester import extract_text_from_pdf, test_rag_prompt

    prompt = request.form.get('prompt', '')
    expected = request.form.get('expected', '')

    if not prompt or not expected:
        return jsonify({'error': 'Prompt and expected behaviour are required'}), 400

    if 'document' not in request.files:
        return jsonify({'error': 'No document uploaded'}), 400

    file = request.files['document']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        upload_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(upload_path)

        if file.filename.endswith('.pdf'):
            document_text = extract_text_from_pdf(upload_path)
        else:
            with open(upload_path, 'r', encoding='utf-16') as f:
                document_text = f.read()

        os.remove(upload_path)

        result = test_rag_prompt(
            prompt=prompt,
            expected_behaviour=expected,
            document_text=document_text,
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/agent', methods=['POST'])
def run_agent():
    """Run the autonomous mutation agent."""
    data = request.json
    prompt = data.get('prompt', '')
    expected = data.get('expected', '')
    test_input = data.get('test_input', '')
    target_score = data.get('target_score', 80.0)
    max_steps = data.get('max_steps', 10)

    if not prompt or not expected:
        return jsonify({'error': 'Prompt and expected behaviour are required'}), 400

    try:
        from agent import run_autonomous_agent
        result = run_autonomous_agent(
            prompt=prompt,
            expected_behaviour=expected,
            test_inputs=[test_input],
            target_score=target_score,
            max_steps=max_steps,
        )
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/export/csv/<int:run_id>')
def export_csv(run_id):
    """Export a test run as CSV."""
    from exporter import export_csv
    from flask import Response

    run = get_run_by_id(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404

    csv_bytes = export_csv(run)

    return Response(
        csv_bytes,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=prompt_test_{run_id}.csv'}
    )


@app.route('/export/pdf/<int:run_id>')
def export_pdf(run_id):
    """Export a test run as PDF."""
    from exporter import export_pdf
    from flask import Response

    run = get_run_by_id(run_id)
    if not run:
        return jsonify({'error': 'Run not found'}), 404

    pdf_bytes = export_pdf(run)

    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={'Content-Disposition': f'attachment; filename=prompt_test_{run_id}.pdf'}
    )


if __name__ == '__main__':
    app.run(debug=True)