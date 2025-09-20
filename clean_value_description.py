#!/usr/bin/env python3
"""
Script to clean value_description columns in CSV files using GPT-4o.
Creates a new column 'value_description_cleaned' with concise, one-sentence descriptions.
"""

import os
import pandas as pd
import openai
from pathlib import Path
import time
import argparse
import json
import asyncio
from typing import List, Tuple

# Configure OpenAI clients
client = openai.OpenAI()
async_client = openai.AsyncOpenAI()

async def clean_description_with_gpt4o_async(description: str, column_name: str = "") -> str:
    """
    Use GPT-4o to clean and concisely rewrite a value description.

    Args:
        description: The original value description text
        column_name: The name of the column for context

    Returns:
        Cleaned, concise description (1-2 sentences max)
    """
    if not description or description.strip() == "":
        return ""

    prompt = f"""Clean and rewrite this database column value description into 1-2 concise sentences while retaining all important information.

Column name: {column_name}
Original description: {description}

Requirements:
- Keep it to 1-2 sentences maximum
- Retain all important technical details and values
- Remove verbose explanations and formatting characters
- common sense evidence is important, do not miss it while cleaning
- Make it clear and professional
- If there are specific codes/values listed, keep them but make the format clean

Provide only the cleaned description, no additional text. Return a json object with the following fields:
- description: the cleaned description without any newlines (single line please, no markdown related stuff)
- rationale: the rationale for the cleaned description
"""

    try:
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a technical writer specializing in database documentation. Your job is to make descriptions concise while preserving all important information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        cleaned = json.loads(response.choices[0].message.content.strip())
        return cleaned['description']

    except Exception as e:
        print(f"Error processing description for column {column_name}: {e}")
        # Fallback: return first sentence if API fails
        first_sentence = description.split('.')[0].strip()
        return first_sentence if first_sentence else description[:100] + "..."

async def process_descriptions_in_batches(rows_to_process: List[Tuple[int, str, str]], batch_size: int = 10) -> List[str]:
    """
    Process descriptions in batches using async concurrency.

    Args:
        rows_to_process: List of (index, description, column_name) tuples
        batch_size: Number of concurrent API calls

    Returns:
        List of cleaned descriptions in the same order as input
    """
    cleaned_descriptions = [""] * len(rows_to_process)

    # Process in batches
    for i in range(0, len(rows_to_process), batch_size):
        batch = rows_to_process[i:i + batch_size]
        batch_start = i + 1
        batch_end = min(i + batch_size, len(rows_to_process))

        print(f"    Processing batch {batch_start}-{batch_end} of {len(rows_to_process)}...")

        # Create async tasks for this batch
        tasks = []
        for idx, description, column_name in batch:
            if description.strip():  # Only process non-empty descriptions
                task = clean_description_with_gpt4o_async(description, column_name)
                tasks.append((idx, task))
            else:
                cleaned_descriptions[idx] = ""

        # Wait for all tasks in this batch to complete
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            # Store results in correct positions
            for (idx, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"      Error processing row {idx + 1}: {result}")
                    # Fallback for errors
                    original_desc = rows_to_process[idx][1]
                    cleaned_descriptions[idx] = original_desc.split('.')[0].strip() if original_desc else ""
                else:
                    cleaned_descriptions[idx] = result

        # Small delay between batches to be respectful to API
        if i + batch_size < len(rows_to_process):
            await asyncio.sleep(0.5)

    return cleaned_descriptions

async def process_csv_file_async(csv_path: Path, dry_run: bool = False, batch_size: int = 10) -> None:
    """
    Process a single CSV file to add cleaned value descriptions.

    Args:
        csv_path: Path to the CSV file
        dry_run: If True, don't save changes, just show what would be done
    """
    print(f"\nProcessing: {csv_path}")

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)

        # Check if value_description column exists
        if 'value_description' not in df.columns:
            print(f"  Skipping: No 'value_description' column found")
            return

        # Check if cleaned column already exists
        if 'value_description_cleaned' in df.columns:
            print(f"  Warning: 'value_description_cleaned' column already exists")
            if not dry_run:
                user_input = input("  Overwrite existing cleaned descriptions? (y/N): ")
                if user_input.lower() != 'y':
                    print("  Skipping file...")
                    return

        # Get column names for context
        column_names = df.get('column_name', df.get('original_column_name', ''))

        # Prepare data for batch processing
        rows_to_process = []
        for idx, row in df.iterrows():
            description = str(row['value_description']) if pd.notna(row['value_description']) else ""
            column_name = str(column_names.iloc[idx]) if pd.notna(column_names.iloc[idx]) else ""
            rows_to_process.append((idx, description, column_name))

        total_rows = len(rows_to_process)
        print(f"  Processing {total_rows} rows in batches of {batch_size}...")

        if dry_run:
            # For dry run, just simulate
            cleaned_descriptions = [f"[DRY RUN] Would clean: {desc[:100]}..." for _, desc, _ in rows_to_process]
        else:
            # Process in batches using async
            cleaned_descriptions = await process_descriptions_in_batches(rows_to_process, batch_size)

        # Add the new column
        df['value_description_cleaned'] = cleaned_descriptions

        if not dry_run:
            # Save the updated CSV
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved updated file with {len(cleaned_descriptions)} cleaned descriptions")
        else:
            print(f"  ✓ Would add {len(cleaned_descriptions)} cleaned descriptions")

    except Exception as e:
        print(f"  ✗ Error processing {csv_path}: {e}")

def find_csv_files(base_path: Path) -> list[Path]:
    """Find all CSV files in the database description directories."""
    csv_files = []

    # Look for database_description directories
    for db_dir in base_path.iterdir():
        if db_dir.is_dir():
            desc_dir = db_dir / "database_description"
            if desc_dir.exists() and desc_dir.is_dir():
                # Find CSV files in this directory
                for csv_file in desc_dir.glob("*.csv"):
                    csv_files.append(csv_file)

    return csv_files

async def main_async():
    parser = argparse.ArgumentParser(description="Clean value descriptions in CSV files using GPT-4o")
    parser.add_argument("--base-path",
                       default="/Users/dsaluru/Desktop/GPU/bird-sql-forked/data/test_databases",
                       help="Base path to search for CSV files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--file", help="Process a specific CSV file instead of searching")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of concurrent API calls (default: 10)")

    args = parser.parse_args()

    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return

    base_path = Path(args.base_path)

    if args.file:
        # Process specific file
        csv_path = Path(args.file)
        if not csv_path.exists():
            print(f"Error: File {csv_path} does not exist")
            return
        await process_csv_file_async(csv_path, args.dry_run, args.batch_size)
    else:
        # Find and process all CSV files
        if not base_path.exists():
            print(f"Error: Base path {base_path} does not exist")
            return

        csv_files = find_csv_files(base_path)

        if not csv_files:
            print(f"No CSV files found in database_description directories under {base_path}")
            return

        print(f"Found {len(csv_files)} CSV files to process:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")

        if not args.dry_run:
            print(f"\nThis will use GPT-4o API calls (approximately {sum(len(pd.read_csv(f)) for f in csv_files)} calls)")
            user_input = input("Continue? (y/N): ")
            if user_input.lower() != 'y':
                print("Cancelled.")
                return

        # Process each file
        for csv_file in csv_files:
            await process_csv_file_async(csv_file, args.dry_run, args.batch_size)

        print(f"\n✓ Completed processing {len(csv_files)} files")

def main():
    """Main entry point that runs the async main function."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()