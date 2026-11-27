"""CLI Entry Point for LitScope-Lite."""

import argparse
import json
import sys
from pathlib import Path
from src.pipeline import SciFactPipeline

def main():
    parser = argparse.ArgumentParser(description="LitScope-Lite: End-to-End Scientific Claim Verification")
    
    # Create mutually exclusive group so user must provide EITHER text OR a file, but not both
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw text containing scientific claim(s) to verify")
    group.add_argument("--file", type=str, help="Path to a JSONL file containing claims")
    
    parser.add_argument("--output", type=str, default="verification_report.json", help="Path to save the JSON output")
    
    args = parser.parse_args()

    # Initialize the pipeline
    try:
        pipeline = SciFactPipeline()
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    final_report = []

    # Handle raw text input
    if args.text:
        print("\nProcessing raw text input...")
        final_report = pipeline.verify(args.text)

    # Handle JSONL file input
    elif args.file:
        print(f"\nProcessing JSONL file: {args.file}")
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
            
        reports = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    claim_text = data.get("claim", "")
                    if claim_text:
                        print(f"Verifying claim {line_idx}...")
                        reports.append(pipeline.verify(claim_text))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_idx}")
        final_report = reports

    # Save the output to a JSON file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\nVerification complete! Detailed report saved to: {args.output}")

if __name__ == "__main__":
    main()