import argparse
import logging
import os
import random
import re

import numpy as np
import sqlglot
from sqlglot import expressions as exp
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sqlalchemy import create_engine

from utils import append_jsonl_file, read_json_file, read_jsonl_file
from schema_engine import SchemaEngine


def extract_schema_linking(sql_query, dialect="sqlite"):
    """Extract table and column information from SQL query using sqlglot."""
    try:
        # Clean up any ANSI escape codes
        clean_query = re.sub(r'\x1b\[[0-9;]*m', '', sql_query)

        parsed = sqlglot.parse_one(clean_query, dialect="sqlite")

        # Build alias mapping
        alias_to_table = {}
        for table in parsed.find_all(exp.Table):
            if table.alias:
                alias_to_table[table.alias] = table.name
            alias_to_table[table.name] = table.name

        schema_linking = {}

        # Find all column references
        for column in parsed.find_all(exp.Column):
            if column.table:
                table_name = alias_to_table.get(column.table, column.table)
                column_name = column.name

                if table_name not in schema_linking:
                    schema_linking[table_name] = set()
                schema_linking[table_name].add(column_name)

        return {table: list(columns) for table, columns in schema_linking.items()}

    except Exception as e:
        return {"error": str(e)}


def get_linked_schema(db_path, db_id, sql_query):
    """Generate schema for only the tables/columns used in the SQL query."""
    try:
        # Extract table/column information from SQL
        schema_info = extract_schema_linking(sql_query)

        if "error" in schema_info:
            # If parsing fails, fall back to full schema
            return open(f"{db_path}.mschema", "r").read()

        # Get the tables used in the query
        tables_used = list(schema_info.keys())

        if not tables_used:
            # If no tables found, fall back to full schema
            return open(f"{db_path}.mschema", "r").read()

        # Create schema engine and generate filtered mschema
        db_engine = create_engine(f"sqlite:///{db_path}")
        schema_engine = SchemaEngine(engine=db_engine, db_name=db_id)

        # Generate mschema for only the linked tables
        linked_mschema = schema_engine.mschema.to_mschema(selected_tables=tables_used)

        return linked_mschema

    except Exception as e:
        logging.warning(f"Schema linking failed for {db_id}: {e}. Using full schema.")
        # Fall back to full schema if anything goes wrong
        return open(f"{db_path}.mschema", "r").read()


# Configure logging
def setup_logging(log_file):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


def generate_prompt_message(datum, use_schema_linking=True):
    if use_schema_linking and "SQL" in datum and datum["SQL"]:
        # Use schema linking with golden SQL
        schema_content = get_linked_schema(datum['db_path'], datum.get('db_id', 'unknown'), datum["SQL"])
    else:
        # Fall back to full schema
        print(f"Using full schema for {datum['db_id']}")
        schema_content = open(f"{datum['db_path']}.mschema", "r").read()

    return (
        schema_content
        + "\n\n"
        + "-- External Knowledge: {}".format(datum["evidence"])
        + "\n"
        + "-- Using valid SQLite and understanding External Knowledge, "
        "answer the following questions for the tables provided above."
        + "\n"
        + "-- {}".format(datum["question"])
        + "\nJust output SQL starting with SELECT directly, dont output anything else."
    )


def generate_fewshot_messages(args, fewshot_data, rng):
    """Randomly sample `args.num_shots` few-shot examples from `fewshot_data`."""
    sampled_indices = rng.sample(list(range(len(fewshot_data))), args.num_shots)
    sampled_indices_str = ", ".join(map(str, sampled_indices))
    logging.debug(f"Sampled few-shot indices: {sampled_indices_str}")

    messages = []
    for i in range(args.num_shots):
        sampled_index = sampled_indices[i]
        messages.append(
            {
                "role": "user",
                "content": generate_prompt_message(fewshot_data[sampled_index], use_schema_linking=False),
            }
        )
        messages.append(
            {"role": "assistant", "content": fewshot_data[sampled_index]["SQL"]}
        )
    logging.info(
        f"Generated {args.num_shots} few-shot messages using indices {sampled_indices_str}"
    )
    return messages, sampled_indices


def generate_prompt_messages(args, datum, fewshot_messages):
    return [
        {"role": "system", "content": args.system_prompt},
        *fewshot_messages,
        {"role": "user", "content": generate_prompt_message(datum, use_schema_linking=args.use_schema_linking)},
    ]


def run_generations_for_batch(
    args,
    data_batch,
    fewshot_messages,
    fewshot_indices,
    tokenizer,
    vllm_model,
    sampling_params,
):
    """Generate for one batch of data (size ~ args.batch_size)."""
    logging.debug(f"Preparing prompts for a batch of {len(data_batch)} items.")
    prompts_batch = []
    prompt_metadata = []

    for i, datum in enumerate(data_batch):
        messages = generate_prompt_messages(args, datum, fewshot_messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_metadata.append(
            {
                "prompt": prompt,
                "source_str": f"|shots{','.join(map(str, fewshot_indices))}|",
                "data_idx": i,  # index within this batch
            }
        )
        prompts_batch.append(prompt)

    logging.debug("Running generation on the batch...")
    outputs = vllm_model.generate(prompts_batch, sampling_params)
    logging.debug("Generation for this batch completed.")

    # Store the results in each datum
    for pm_i, out_i in zip(prompt_metadata, outputs):
        datum_idx = pm_i["data_idx"]
        datum = data_batch[datum_idx]
        datum["responses"] = {}
        for response_idx, choice in enumerate(out_i.outputs):
            generated_sql = choice.text.strip()
            if generated_sql not in datum["responses"]:
                datum["responses"][generated_sql] = {
                    "content": generated_sql,
                    "sources": [f"{pm_i['source_str']}{response_idx}"],
                    "all_logprobs": [choice.cumulative_logprob],
                }
            else:
                existing = datum["responses"][generated_sql]
                existing["sources"].append(f"{pm_i['source_str']}{response_idx}")
                existing["all_logprobs"].append(choice.cumulative_logprob)

    return data_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--few_shot_dir", type=str, default="few_shots", required=False)
    parser.add_argument(
        "--model", type=str, default="models/generator", required=False
    )
    parser.add_argument("--system_prompt", type=str, default="You are a SQLite expert tasked with writing SQL for a given natural language user query. You would be given database information in form of CREATE TABLE statements; External Knowledge which are hints; user natural language query. Your task is to write valid SQLite to answer the user questions for the tables provided.", required=False)
    parser.add_argument("--batch_size", type=int, default=256, required=False)
    parser.add_argument("--num_samples_per_prompt", type=int, default=16, required=False)
    parser.add_argument("--num_prompts_per_query", type=int, default=16, required=False)
    parser.add_argument("--num_shots", type=int, default=1, required=False)
    parser.add_argument("--max_tokens", type=int, default=1024, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--temperature", type=float, default=1.0, required=False)
    parser.add_argument("--num_gpus", type=int, default=1, required=False)
    parser.add_argument(
        "--start_from_scratch", dest="start_from_scratch", action="store_true"
    )
    parser.add_argument(
        "--stop",
        type=str,
        default="; <pad> <|endoftext|> </s> <|im_end|>",
        required=False,
    )
    parser.add_argument("--num_partitions", type=int, default=1, required=False)
    parser.add_argument("--partition_index", type=int, default=0, required=False)
    parser.add_argument("--use_schema_linking", action="store_true", help="Use schema linking to filter tables based on golden SQL")

    args = parser.parse_args()
    args.stop = args.stop.split()

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.output_dir, "generation.log")
    setup_logging(log_file)

    # Log command-line arguments
    logging.info("Starting generation script with arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Read data
    logging.info("Reading input data...")
    data = read_jsonl_file(args.input_file)
    data = np.array_split(data, args.num_partitions)[args.partition_index].tolist()
    logging.info(
        f"Data size after partitioning: {len(data)} items "
        f"(partition {args.partition_index + 1} of {args.num_partitions})"
    )

    # Read few-shot examples
    fewshot_data = read_json_file(os.path.join(args.few_shot_dir, "data.json"))
    for datum in fewshot_data:
        datum['db_path'] = os.path.join(args.few_shot_dir, 'mschema', datum['db_id'])
    logging.info(f"Read {len(fewshot_data)} few-shot examples from {args.few_shot_dir}")

    # Prepare output directory
    out_file_name = (
        f"data_gen_{args.partition_index}-of-{args.num_partitions}.jsonl"
        if args.num_partitions > 1
        else "data_gen.jsonl"
    )
    out_file_path = os.path.join(args.output_dir, out_file_name)
    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
    logging.info(f"Output file path: {out_file_path}")

    # Restart mechanism
    progress_file = out_file_path + ".progress"

    if args.start_from_scratch:
        logging.info("--start_from_scratch flag is set.")
        if os.path.exists(progress_file):
            logging.info(f"Deleting existing progress file at {progress_file}.")
            os.remove(progress_file)

    start_iteration = 0
    start_data_idx = 0
    if not args.start_from_scratch and os.path.exists(progress_file):
        with open(progress_file, "r") as pf:
            line = pf.read().strip()
            if line:
                start_iteration, start_data_idx = map(int, line.split(","))
                logging.info(
                    f"Resuming from iteration {start_iteration}, data index {start_data_idx}"
                )
    else:
        logging.info("No existing progress to resume from. Starting from scratch.")

    # Initialize tokenizer & model
    logging.info("Initializing model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    vllm_model = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=16384,
        tensor_parallel_size=args.num_gpus,
        swap_space=0,
    )
    logging.info("Model and tokenizer initialized.")

    sampling_params = SamplingParams(
        n=args.num_samples_per_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=args.stop,
        logprobs=1,
    )

    rng = random.Random(args.seed)

    # Fast-forward the random state for iterations we've already done
    logging.debug(f"Fast-forwarding random seed for {start_iteration} iterations...")
    for _ in range(start_iteration):
        _ = generate_fewshot_messages(args, fewshot_data, rng)
    logging.debug("Done fast-forwarding random state.")

    # Main generation loop
    for iteration_index in range(start_iteration, args.num_prompts_per_query):
        logging.info(
            f"=== Starting iteration {iteration_index} / {args.num_prompts_per_query - 1} ==="
        )
        fewshot_messages, fewshot_indices = generate_fewshot_messages(
            args, fewshot_data, rng
        )

        batch_start = start_data_idx
        while batch_start < len(data):
            batch_end = min(batch_start + args.batch_size, len(data))
            logging.info(
                f"Generating batch from data index {batch_start} to {batch_end - 1}..."
            )
            data_batch = data[batch_start:batch_end]

            data_batch = run_generations_for_batch(
                args,
                data_batch,
                fewshot_messages,
                fewshot_indices,
                tokenizer,
                vllm_model,
                sampling_params,
            )

            # Append results to output file
            append_jsonl_file(out_file_path, data_batch)
            logging.info(f"Appended {len(data_batch)} items to {out_file_path}")

            batch_start += len(data_batch)

            # Save progress
            with open(progress_file, "w") as pf:
                pf.write(f"{iteration_index},{batch_start}")
            logging.info(
                f"Progress saved: iteration={iteration_index}, next data index={batch_start}"
            )

        # Reset start_data_idx for next iteration
        start_data_idx = 0
        logging.info(f"Completed iteration {iteration_index}")

    logging.info("All generations completed!")
    logging.info("Exiting script.")


if __name__ == "__main__":
    main()
