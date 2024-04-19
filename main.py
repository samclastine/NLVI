from Experiments import VegaLiteEvaluator_EX1A
import argparse
import logging
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Run experiments for data evaluation")
    parser.add_argument('--exp', type=str, help='Experiment identifier (e.g., ex1a, ex1b)')
    parser.add_argument('--output', type=str, help='Output CSV file name')
    parser.add_argument('--modelID', type=str, help='Model identifier')
    return parser


def run_experiment(exp_name, result_filename, model_id):
    if exp_name=='ex1a':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX1A(model_id=model_id, output_filename=result_filename)
        queries_df = pd.read_csv('/content/NLVI/eval_data/nlvCorpus_150.csv')
        result = evaluator.run_evaluation(queries_df)
        print(result)
    if exp_name=='ex1b':
        logging.info(f"Running {exp_name} with model {model_id}")
        VegaLiteEvaluator_EX1A(llm=llm, output_filename=result_filename)
        



def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    print("im here")

    if args.exp and args.output and args.modelID:
      print("helllooooooooo im here")
      run_experiment(args.exp, args.output, args.modelID)
    else:
        print("Missing arguments, please specify --exp, --output, and --modelID.")

if __name__ == "__main__":
    main()




