
# Running the Main Script

This README outlines the steps to run the `main.py` script from the NLVI project. The script supports various experimental setups and utilizes models from Hugging Face.

## Prerequisites

Ensure that the NLVI repository is cloned to your local machine and that all dependencies listed in the `requirements.txt` file are installed.

## How to Run the Script

The script can be executed with different parameters to specify the experiment, model, and output location. Here are the steps:

1. **Open your command line interface**.
2. **Navigate to the directory containing `main.py`**.
3. **Run the script with the desired parameters**:

```bash
python main.py --exp "<experiment_name>" --modelID "<hugging_face_model_id>" --output "<output_file_path>"
```

### Experiment Options

You can replace `<experiment_name>` with any of the following to specify the experiment you wish to run:
- `ex1a [Evaluating without Data in Prompt - Zero Shot]`
- `ex1b [Evaluating with Data in Prompt - Zero Shot]`
- `ex2a [Few Shot (generic examples)]`
- `ex2b [Few shot COT (generic examples)]`
- `ex3a [RAG One Shot]`
- `ex3b [RAG One Shot with CoT]`
- `ex4a [Zero Shot Baseline]`
- `ex4b [Zero Shot CoT]`
- `ex5a [CoT Baseline]`

### Model Options

Replace `<hugging_face_model_id>` with any compatible model ID from Hugging Face that suits your experiment.

### Output Options

The output file path should be specified in the command, and the script only supports output in CSV format. Ensure the path ends with `.csv`.

## Example

Here is an example command to run the script for experiment `ex1a` with a specific Hugging Face model, outputting results to `output.csv`:

```bash
python main.py --exp "ex1a" --modelID "deepseek-ai/deepseek-coder-1.3b-instruct" --output "output.csv"
```
