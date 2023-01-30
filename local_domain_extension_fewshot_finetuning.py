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

    full_conll = get_corpus(name="conll", map="short", path=args.cache_path)
    average_over_support_sets = []
    for split in range(args.splits):
        try:
            tars_tagger: FewshotClassifier = TARSTagger.load(
                f"{args.cache_path}/flair-models/pretrained-few-shot/{args.transformer}_{args.pretraining_corpus}_{args.lr}-{args.seed}/final-model.pt")
        except:
            raise FileNotFoundError(
                f"{args.cache_path}/flair-models/pretrained-few-shot/{args.transformer}_{args.pretraining_corpus}_{args.lr}-{args.seed}/final-model.pt - has this model been trained?")

        support_set = ColumnCorpus(
            data_folder=f"data/fewshot/{args.fewshot_corpus}/{args.k}shot",
            train_file=f"{split}.txt",
            sample_missing_splits=False,
            column_format={0: "text", 1: "ner"},
            label_name_map=get_label_name_map(args.fewshot_corpus))
        print(support_set)

        dictionary = support_set.make_label_dictionary('ner')
        print(dictionary)

        tars_tagger.add_and_switch_to_new_task(task_name="fewshot-conll-short",
                                           label_dictionary=dictionary,
                                           label_type="ner",
                                           force_switch=True)

        trainer = ModelTrainer(tars_tagger, support_set)

        trainer.fine_tune(
            f'{args.cache_path}/flair-models/finetuned-few-shot/{args.transformer}_{args.pretraining_corpus}_{args.fewshot_corpus}_{args.lr}-{args.seed}/split_{split}',
            learning_rate=args.lr,
            mini_batch_size=args.bs,
            mini_batch_chunk_size=args.mbs,
            train_with_dev=False,
            monitor_test=False,
            save_final_model=False,
            max_epochs=50
            )

        result = tars_tagger.evaluate(data_points=full_conll.test, gold_label_type="ner")
        with open(f'{args.cache_path}/flair-models/finetuned-few-shot/{args.transformer}_{args.fewshot_corpus}_{args.lr}-{args.seed}/split_{split}/result.txt', "w") as f:
            f.write(result.detailed_results)

        average_over_support_sets.append(result.main_score)

    with open(
            f'{args.cache_path}/flair-models/finetuned-few-shot/{args.transformer}_{args.fewshot_corpus}_{args.lr}-{args.seed}/average_result.txt',
            "w") as f:
        f.write(f"average micro f1: {(sum(average_over_support_sets) / len(average_over_support_sets))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_path", type=str, default="/glusterfs/dfs-gfs-dist/goldejon")
    parser.add_argument("--pretraining_corpus", type=str, default="ontonotes")
    parser.add_argument("--fewshot_corpus", type=str, default="conll03")
    parser.add_argument("--transformer", type=str, default="xlm-roberta-large")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--mbs", type=int, default=2)
    args = parser.parse_args()
    main(args)
