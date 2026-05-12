#!/usr/bin/env python3
"""
Development Benchmark Creator

Creates a small subset of LongBench for fast iteration during development.
"""

from datasets import load_dataset
import argparse

def create_dev_benchmark(output_dir: str, num_samples: int = 50):
    """
    Create a small development benchmark from LongBench.

    Args:
        output_dir: Directory to save the dev benchmark
        num_samples: Number of samples to include
    """
    print(f"Loading LongBench dataset...")

    # LongBench has multiple tasks, we'll sample from each
    tasks = [
        "longdoc_qa_eng",
        "longdoc_summarization_eng",
        "longdoc_qa",
        "longdoc_summarization",
        "few_shot_qa_eng",
        "few_shot_qa",
        "code_completion",
    ]

    try:
        # Try loading the full LongBench dataset
        ds = load_dataset("THUDM/LongBench", split="test")
        print(f"Full dataset has {len(ds)} examples")

        # Select first num_samples for dev
        dev_data = ds.select(range(min(num_samples, len(ds))))
        dev_data.save_to_disk(output_dir)
        print(f"Saved {len(dev_data)} examples to {output_dir}")

        # Print sample info
        print("\nSample structure:")
        print(dev_data[0].keys())

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dev benchmark instead...")

        # Create synthetic benchmark if dataset fails to load
        import json
        import os

        os.makedirs(output_dir, exist_ok=True)

        synthetic_data = [
            {
                "input": "Summarize the following text in 3 sentences: " +
                         "Artificial intelligence has revolutionized many industries. " +
                         "Machine learning algorithms can process vast amounts of data quickly. " +
                         "Deep learning models have achieved remarkable results in image and text processing. " +
                         "However, ethical considerations remain important in AI development. " +
                         "The future of AI holds great promise for solving complex problems.",
                "output": "AI has revolutionized industries through ML algorithms and deep learning. " +
                          "Ethical considerations remain important in development. " +
                          "The future holds promise for solving complex problems.",
                "task_type": "summarization",
                "context_length": 100,
            }
        ] * num_samples

        # Save as JSON
        json_path = os.path.join(output_dir, "dev_benchmark.json")
        with open(json_path, 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        print(f"Created synthetic benchmark at {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dev benchmark")
    parser.add_argument("--output", default="data/longbench_dev_50", help="Output directory")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")

    args = parser.parse_args()
    create_dev_benchmark(args.output, args.samples)