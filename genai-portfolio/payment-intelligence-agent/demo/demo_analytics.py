"""Standalone analytics demo - NL-to-SQL payment query translation.

Demonstrates the full analytics pipeline: natural language query parsing,
SQL generation, safety validation, result generation (synthetic), and
natural language summarization.

Run: python -m demo.demo_analytics
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.nl_to_sql import NLToSQLTranslator
from src.analytics.query_validator import QueryValidator
from src.analytics.result_analyzer import ResultAnalyzer


def main() -> None:
    translator = NLToSQLTranslator()
    validator = QueryValidator()
    analyzer = ResultAnalyzer()

    queries = [
        "Show me top 10 merchants by transaction volume last month",
        "What is the average settlement time by payment method?",
        "How many chargebacks occurred last month?",
        "Compare revenue across regions",
        "What is the decline rate by payment method?",
        "Show me the daily transaction volume trend",
        "Which merchants have the highest chargeback rate?",
        "Show me the payment method breakdown",
    ]

    print("=" * 80)
    print("  PAYMENT INTELLIGENCE AGENT - Analytics Demo")
    print("  NL-to-SQL Translation Pipeline")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"  Query {i}: {query}")
        print(f"{'─' * 80}")

        # Translate to SQL
        result = translator.translate(query)

        print(f"\n  Template: {result['template'] or 'custom'}")
        print(f"  Time Range: {result['time_range']}")
        print(f"  Explanation: {result['explanation']}")
        print(f"\n  Generated SQL:")
        for line in result["sql"].strip().split("\n"):
            print(f"    {line.strip()}")

        # Validate SQL
        validation = validator.validate(result["sql"])
        print(f"\n  Validation: {'PASS' if validation.is_safe else 'FAIL'}")
        if validation.warnings:
            for w in validation.warnings:
                print(f"    Warning: {w}")

        # Generate demo results
        df = translator.generate_demo_results(query, result["sql"])
        print(f"\n  Results: {len(df)} rows x {len(df.columns)} columns")
        print(f"  Columns: {', '.join(df.columns.tolist())}")

        if not df.empty:
            print(f"\n  Sample data:")
            print(df.head(5).to_string(index=False, max_cols=6))

            # Generate summary
            summary = analyzer.summarize(df, query)
            print(f"\n  Summary: {summary}")

    print(f"\n{'=' * 80}")
    print("  Demo complete. All queries translated, validated, and executed.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
