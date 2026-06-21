from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

# Create the database instance
db = SQLAlchemy()


class TestRun(db.Model):
    """Stores every prompt test run."""

    __tablename__ = 'test_runs'

    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.Text, nullable=False)
    expected_behaviour = db.Column(db.Text, nullable=False)
    test_input = db.Column(db.Text, nullable=False)
    overall_score = db.Column(db.Float, nullable=False)
    verdict = db.Column(db.String(20), nullable=False)
    fixed_prompt = db.Column(db.Text, nullable=True)
    strategy_scores = db.Column(db.Text, nullable=False)  # stored as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary for JSON response."""
        return {
            'id': self.id,
            'prompt': self.prompt,
            'expected_behaviour': self.expected_behaviour,
            'test_input': self.test_input,
            'overall_score': self.overall_score,
            'verdict': self.verdict,
            'fixed_prompt': self.fixed_prompt,
            'strategy_scores': json.loads(self.strategy_scores),
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
        }


def save_test_run(
    prompt: str,
    expected_behaviour: str,
    test_input: str,
    overall_score: float,
    strategy_scores: dict,
    fixed_prompt: str = None,
) -> TestRun:
    """Save a test run to the database."""

    # Determine verdict from score
    if overall_score >= 80:
        verdict = 'ROBUST'
    elif overall_score >= 55:
        verdict = 'MODERATE'
    else:
        verdict = 'FRAGILE'

    # Create and save the record
    run = TestRun(
        prompt=prompt,
        expected_behaviour=expected_behaviour,
        test_input=test_input,
        overall_score=overall_score,
        verdict=verdict,
        fixed_prompt=fixed_prompt,
        strategy_scores=json.dumps(strategy_scores),
    )

    db.session.add(run)
    db.session.commit()

    return run


def get_all_runs() -> list:
    """Get all test runs ordered by most recent first."""
    runs = TestRun.query.order_by(TestRun.created_at.desc()).all()
    return [r.to_dict() for r in runs]


def get_run_by_id(run_id: int) -> dict:
    """Get a single test run by ID."""
    run = TestRun.query.get(run_id)
    return run.to_dict() if run else None