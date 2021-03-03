import flair
from flair.data import Corpus, MultitaskCorpus
from flair.datasets import CSVClassificationCorpus, TREC_50, TREC_6, AMAZON_REVIEWS
from flair.models.text_classification_model import TARSClassifier
from flair.models.multitask_model import MultitaskModel
from flair.trainers import ModelTrainer

def main():
    trec6_label_name_map = {'ENTY': 'question about entity',
                            'DESC': 'question about description',
                            'ABBR': 'question about abbreviation',
                            'HUM': 'question about person',
                            'NUM': 'question about number',
                            'LOC': 'question about location'
                            }
    trec6_corpus: Corpus = TREC_6(label_name_map=trec6_label_name_map)
    trec6_labels = trec6_corpus.make_label_dictionary()

    tars = TARSClassifier(task_name='TREC_6', label_dictionary=trec6_labels, document_embeddings='bart-base-mnli')

    trainer = ModelTrainer(tars, trec6_corpus)
    trainer.train(base_path='resources/taggers/tars',  # path to store the model artifacts
                  learning_rate=0.02,  # use very small learning rate
                  mini_batch_size=16,
                  max_epochs=20,  # terminate after 10 epochs
                  )

if __name__ == "__main__":
    main()