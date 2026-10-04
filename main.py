"""Multi-Agent Research Assistant — fully local, no API keys.

Usage:
    python main.py "Latest advances in quantum computing"
    python main.py "Impact of AI on healthcare" --output report.md
"""

import os, argparse
from datetime import datetime
from src.graph import run_pipeline


def main(topic, output_path=None):
    print(f"\n{'='*60}")
    print(f"🔬 Multi-Agent Research Assistant (Local)")
    print(f"Topic: {topic}")
    print(f"{'='*60}")

    state = run_pipeline(topic)
    report = state["final_report"]

    os.makedirs("reports", exist_ok=True)
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c == " " else "_" for c in topic)[:50]
        output_path = f"reports/{safe.strip().replace(' ', '_')}_{ts}.md"

    with open(output_path, "w") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(report)

    print(f"\n{'='*60}")
    print(f"📄 Saved: {output_path}")
    print(f"{'='*60}\n")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic", type=str)
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()
    main(args.topic, args.output)
