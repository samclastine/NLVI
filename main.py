from models.HF_Models import initialize_evllm
from Experiments import ex1a, ex1b
import argparse
import logging



def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Run experiments for data evaluation")
    parser.add_argument('--exp', type=int, help='Experiment number (1-8)')
    parser.add_argument('--output', type=str, help='Output CSV file name')
    parser.add_argument('--modelID', type=str, help='Model identifier')
    return parser


def run_experiment(exp_name, result_filename, model_id):
    llm = initialize_evllm(model_id=model_id, temperature=0.5)
    if exp_name=='ex1a':
        logging.info(f"Running {exp_name} with model {model_id}")
        ex1a.VegaLiteEvaluator(llm=llm, output_filename=result_filename)
    if exp_name=='ex1b':
        logging.info(f"Running {exp_name} with model {model_id}")
        ex1a.VegaLiteEvaluator(llm=llm, output_filename=result_filename)
        



def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.exp and args.output and args.modelID:
        run_experiment(args.exp, args.output, args.modelID)
    else:
        print("Missing arguments, please specify --exp, --output, and --modelID.")

if __name__ == "__main__":
    main()




