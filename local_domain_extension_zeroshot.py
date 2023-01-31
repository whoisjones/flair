import argparse
from flair.models import TARSTagger, FewshotClassifier
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
import flair

from local_domain_extension_label_name_map import get_corpus, get_label_name_map

def main(args):
    flair.set_seed(args.seed)

    if args.cuda:
        flair.device = f"cuda:{args.cuda_device}"

    full_evaluation_dataset = get_corpus(name=args.fewshot_corpus, map="short", path=args.cache_path)
    dictionary = full_evaluation_dataset.make_label_dictionary('ner')
    print(dictionary)

    try:
        tars_tagger: FewshotClassifier = TARSTagger.load(
            f"{args.cache_path}/flair-models/pretrained-few-shot/{args.transformer}_{args.pretraining_corpus}_{args.lr}-{args.seed}/final-model.pt")
    except:
        raise FileNotFoundError(
            f"{args.cache_path}/flair-models/pretrained-few-shot/{args.transformer}_{args.pretraining_corpus}_{args.lr}-{args.seed}/final-model.pt - has this model been trained?")

    tars_tagger.add_and_switch_to_new_task(task_name="zeroshot-short",
                                       label_dictionary=dictionary,
                                       label_type="ner")

    result = tars_tagger.evaluate(data_points=full_evaluation_dataset.test, gold_label_type="ner",
                                  out_path=f'{args.cache_path}/flair-models/finetuned-few-shot/{args.transformer}_{args.pretraining_corpus}_{args.fewshot_corpus}_{args.lr}-{args.seed}/0shot/tart_predict_zeroshot.tsv')
    with open(f'{args.cache_path}/flair-models/finetuned-few-shot/{args.transformer}_{args.pretraining_corpus}_{args.fewshot_corpus}_{args.lr}-{args.seed}/0shot/result.txt', "w") as f:
        f.write(result.detailed_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon")
    parser.add_argument("--pretraining_corpus", type=str, default="ontonotes")
    parser.add_argument("--fewshot_corpus", type=str, default="conll03")
    parser.add_argument("--transformer", type=str, default="xlm-roberta-large")
    args = parser.parse_args()
    main(args)
