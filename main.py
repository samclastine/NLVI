from Experiments import VegaLiteEvaluator_EX1A, VegaLiteEvaluator_EX1B,VegaLiteEvaluator_EX2A, VegaLiteEvaluator_EX3A, VegaLiteEvaluator_EX3B, VegaLiteEvaluator_EX4A, VegaLiteEvaluator_EX4B, VegaLiteEvaluator_EX5 
import argparse
import logging
import warnings
import pandas as pd
import os
import config

warnings.filterwarnings('ignore')
nlvCorpus = pd.read_csv('/content/NLVI/eval_data/nlvCorpus_150.csv')

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Run experiments for data evaluation")
    parser.add_argument('--exp', type=str, help='Experiment identifier (e.g., ex1a, ex1b)')
    parser.add_argument('--output', type=str, help='Output CSV file name')
    parser.add_argument('--modelID', type=str, help='Model identifier')
    parser.add_argument('--openaiAPI', type=str, help='openai api authentication code')
    parser.add_argument('--model_dir', type=str, help='Folder directory for the model has to save')
    parser.add_argument('--JsonEnforcer', type=str, help="JSON Enforcer value has to set 'True' or 'False'")
    parser.add_argument('--mode', type=str, help='value has to set "openai" or "hf')
    return parser


def run_experiment(exp_name, result_filename, model_id, mode, JsonEnforcer):
    if exp_name=='ex1a':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX1A(model_id=model_id, output_filename=result_filename, mode=mode, JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
    if exp_name=='ex1b':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX1B(model_id=model_id, output_filename=result_filename, mode=mode, JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
    if exp_name=='ex2a':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX2A(model_id=model_id, output_filename=result_filename, mode=mode , JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
    if exp_name=='ex3a':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX3A(model_id=model_id, output_filename=result_filename, mode=mode, JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
    if exp_name=='ex3b':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX3B(model_id=model_id, output_filename=result_filename, mode=mode, JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
    if exp_name=='ex4a':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX4A(model_id=model_id, output_filename=result_filename, mode=mode, JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
    if exp_name=='ex4b':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX4B(model_id=model_id, output_filename=result_filename, mode=mode, JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
    if exp_name=='ex5':
        logging.info(f"Running {exp_name} with model {model_id}")
        evaluator = VegaLiteEvaluator_EX5(model_id=model_id, output_filename=result_filename, mode=mode, JsonEnforcer= JsonEnforcer)
        result = evaluator.run_evaluation(nlvCorpus)
        print(result)
        



def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    if args.exp and args.output and args.modelID and args.openaiAPI and args.model_dir and args.mode and args.JsonEnforcer:
      os.environ["OPENAI_API_KEY"] = args.openaiAPI
      config.model_dir = args.model_dir
      run_experiment(args.exp, args.output, args.modelID, args.mode, args.JsonEnforcer)
    else:
        print("Missing arguments, please specify --exp, --output, --openaiAPI and --modelID.")

if __name__ == "__main__":
    main()




