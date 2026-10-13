import modal

# Define the Modal app
app = modal.App("ssm-bench")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Install git for cloning the repository
    .pip_install(
        "torch",
        "matplotlib",
        "numpy",
    )
)


@app.function(
    image=image,
    gpu="A100",  # or "H100" if available
    timeout=3600,  # 1 hour timeout
)
def run_benchmarks():
    import os
    import subprocess
    import sys

    # Clone the repository
    subprocess.run(
        ["git", "clone", "https://github.com/Continum-Computing/ssm-bench.git"], check=True
    )

    os.chdir("ssm-bench")

    # Run the benchmarks
    subprocess.run(
        [
            sys.executable,
            "profiling/run_benchmarks.py",
            "--seq-lens",
            "512",
            "1024",
            "2048",
            "4096",
            "--state-dims",
            "64",
            "128",
            "256",
            "--input-dim",
            "256",
            "--output-dim",
            "256",
            "--dtype",
            "bf16",
            "--device",
            "0",
            "--output",
            "modal_benchmark_results.json",
        ],
        check=True,
    )

    # Read the results
    with open("modal_benchmark_results.json") as f:
        results = f.read()

    return results


@app.local_entrypoint()
def main():
    print("Starting Modal benchmarks on A100...")
    results = run_benchmarks.remote()

    # Save results locally
    with open("modal_results.json", "w") as f:
        f.write(results)

    print("Benchmarks complete! Results saved to modal_results.json")
