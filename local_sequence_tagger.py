from flair.datasets import WNUT_17
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

def main():
    # 1. get the corpus
    corpus = WNUT_17()
    print(corpus)

    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
    print(label_dict)

    # 4. initialize embedding stack with Flair and GloVe
    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),
    ]

    embeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type=label_type,
                            use_crf=True)

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/sota-ner-flair',
                  learning_rate=0.1,
                  mini_batch_size=32,
                  max_epochs=2)

if __name__ == "__main__":
    main()