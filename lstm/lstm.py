from gensim.models.word2vec import Word2Vec # make use of pretrained embeddings
from sklearn.cross_validation import StratifiedKFold
import theano
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot, base_filter, Tokenizer
from keras.utils import np_utils # for converting labels vectors to matrices in multi-class cases
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.callbacks import EarlyStopping
try:
    from keras.utils.visualize_util import model_to_dot, plot
except:
    print("Can't import graphviz-based plotting utilities")
from evaluate import *
from utils import *
import yaml
import pandas as pd
import numpy as np
np.random.seed(42)  # for reproducibility


class Experiment(object):

    def __init__(self, config):
        self.config = config
        # load df
        self.df = self.get_annotations_df()
        # create tokenizer
        self.tokenizer = self.make_tokenizer()
        self.max_features = len(self.tokenizer.word_index) + 1
        self.w2v, self.embedding_weights = self.create_embeddings_weights()
        self.hidden_size = self.w2v.vector_size

    def get_annotations_df(self):
        """
        Remove bugs and empty annotations
        """
        data = pd.read_json(self.config["annotations_file"])
        return data[(data.relation != "Bug") & (data.relation != "")]

    def create_embeddings_weights(self):
        config = self.config
        tk = self.tokenizer
        word2index = tk.word_index
        # reverse index
        index2word = {i:w for (w,i) in tk.word_index.items()}
        max_size = len(index2word) + 1
        # load w2v model
        w2v_vectors_file = config["w2v_data"]
        w2v = Word2Vec.load_word2vec_format(w2v_vectors_file, binary=True)
        word_vector_dims = w2v.vector_size
        embedding_weights = np.zeros((max_size, word_vector_dims))

        for i,w in index2word.items():
            try:
                embedding_weights[i,:] = w2v[w]
            except:
                print("{} not found".format(w))
        return (w2v, embedding_weights)


    def make_tokenizer(self):
        """
        """
        config = self.config
        tk = Tokenizer(
            # the maximum num. of words to retain
            nb_words=None,
            # the characters to filter out from the text
            filters=config["custom_filter"],
            # whether or not to convert the text to lowercase
            lower=True,
            # the character to split on
            split=" ",
            # whether or not to treat each character as a word
            char_level=False
        )
        data = self.df
        x = data.text.values
        # build tokenizer's vocabulary index
        tk.fit_on_texts(x)
        return tk

    def prepare_text(self, x):
        tk = self.tokenizer
        # prepare text
        x = tk.texts_to_sequences(x)
        # pad sequences
        max_len = config["max_len"]
        x = sequence.pad_sequences(x, maxlen=max_len)
        return x

    def prepare_labels(self):
        data = self.df
        # set labels other than precedence to "None"
        label_to_value = config["label_LUT"]
        # filter out bugs and empty relations
        labels = data.relation.replace(label_to_value).values
        return labels

    # folds are made by preserving the percentage of samples for each class
    def prepare_data(self):
        """
        Load annotations from .json,
        discard bugs,
        and replace relations with the relevant classes
        """
        config = self.config
        data = self.df
        # prep text serving as input
        x = self.prepare_text(data.text.values)
        # relations as labels
        labels = self.prepare_labels()
        # find number of classes
        num_classes = len(set(labels))
        return (x, labels)

    def create_model(self):
        """
        """
        config = self.config
        use_pretrained_embeddings = config["with_pretraining"]
        model = Sequential()
        pretrained_embeddings = [self.embedding_weights] if use_pretrained_embeddings else None
        print("Using pretrained embeddings? {}".format(pretrained_embeddings != None))
        max_features = self.max_features
        hidden_size = self.hidden_size
        max_len = config["max_len"]
        num_classes = config["num_classes"]

        # build the embeddings layer
        embeddings = Embedding(
            input_dim=max_features,
            output_dim=hidden_size,
            input_length=max_len,
            W_regularizer=None,
            #weights=None,
            # use pretrained vectors
            weights=pretrained_embeddings,
            dropout=0.2
        )
        model.add(embeddings)
        # build the lstm layer
        lstm = LSTM(
            #input_dim=max_features,
            output_dim=hidden_size,
            dropout_W=0.2,
            dropout_U=0.2,
            return_sequences=False
        )
        model.add(lstm)
        model.add(Dropout(0.5))
        # size should be equal to the number of classes
        model.add(Dense(num_classes))
        # at the end of the day, we only want one label per input (hence softmax)
        model.add(Activation('softmax'))

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=["accuracy"]
        )
        return model

    def write_model_graph(self):
        model = self.create_model()
        # write dot text to file
        with open(config["model_dot"], "wb") as out:
            m = model_to_dot(model)
            dot_text = m.create_dot()
            out.write(dot_text)
            print("Wrote .dot to {}".format(config["model_dot"]))
        # write graph to file
        plot(model, to_file=config["model_graph"], show_shapes=True)

    def train_and_evaluate_model(
        self,
        model,
        x,
        labels,
        train_indices,
        test_indices,
        dev_indices
        ):
        """
        """
        config = self.config
        num_epochs = config["num_epochs"]
        batch_size = config["batch_size"]
        #validation_split = config["validation_split"]
        num_classes = config["num_classes"]

        # convert class vectors to binary class matrices
        labels = np_utils.to_categorical(labels, num_classes)

        # prepare training data
        train_x = x[train_indices]
        train_labels = labels[train_indices]

        # prepare test data
        test_x = x[test_indices]
        test_labels = labels[test_indices]

        # prepare validation dataset
        dev_x = x[dev_indices]
        dev_labels = labels[dev_indices]
        # add early stopping to help avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)

        # train
        model.fit(
            # input
            x=train_x,
            # target labels
            y=train_labels,
            # how many examples to consider at once
            batch_size=batch_size,
            # the number of epochs to train
            nb_epoch=num_epochs,
            # 0 for no logging, 1 for progress bar logging, 2 for one log line per epoch
            verbose=1,
            # the validation data to use,
            validation_data=(dev_x, dev_labels),
            # how much data to reserve for validation (takes n% starting at the end of the dataset)
            #validation_split=0.25,
            # should the training data be shuffled?
            shuffle=True,
            # dict mapping classes to weight for scaling in loss function
            class_weight=None,
            callbacks=[early_stopping]
        )

        # evaluate
        test_predictions = model.predict_classes(test_x, batch_size=batch_size, verbose=0)

        def convert_predictions(predictions):
            """
            converts values in a numpy array to their corresponding label
            """
            value_to_label_LUT = config["value_LUT"]
            for p in predictions:
                yield value_to_label_LUT.get(p, "None")

        test_predictions = list(convert_predictions(test_predictions))
        return test_predictions

    def write_predictions_to_file(self, predictions):
        clf_results = config["classifier_results"]
        # load gold
        gold = self.get_gold_labels()
        df = pd.DataFrame({"Gold":gold, "Predicted":predictions})
        df.to_csv(config["classifier_results"], sep="\t")

    def evaluate(self):
        """
        """
        clf_results = config["classifier_results"]

        evaluator = Evaluator(clf_results)
        classifier_performance = evaluator.generate_scores_df()

        print("Classifier performance")
        print(classifier_performance.round(2))
        print()


    def get_gold_labels(self):
        annotations_path = config["annotations_file"]
        # set labels other than precedence to "None"
        # value -> label
        label_to_value = config["label_LUT"]
        value_to_label = config["value_LUT"]
        data = self.df
        gold_labels = data.relation.replace(label_to_value)
        gold_labels = gold_labels.replace(value_to_label)
        return gold_labels.values

    def run_kfold(self):
        tk = self.tokenizer
        w2v = self.w2v
        max_features = self.max_features
        print("Max features: {}".format(max_features))
        # the number of hidden units
        folds = config["folds"]
        gold = self.get_gold_labels()
        # get text and labels
        # includes preprocessing of text (tokenize, etc.)
        x, labels = self.prepare_data()
        print("# gold labels: {}".format(len(gold)))
        print("max # epochs: {}".format(config["num_epochs"]))
        skf = StratifiedKFold(labels, n_folds=folds, shuffle=True)

        predictions = dict()
        skf = list(skf)
        for i, (train, test) in enumerate(skf):
            print("Running fold {} / {} ...".format(i+1, folds))
            model = None
            model = self.create_model()
            # use next fold for validation
            if i == 0 and i + 1 < len(skf):
                (dev, _) = skf[i+1]
            else:
                (dev, _) = skf[i-1]
            test_predictions = self.train_and_evaluate_model(model, x, labels, train, test, dev)
            # store predictions
            for i in range(len(test)):
                # check each test index
                test_index = test[i]
                # the ith item in the test_predictions corresponds to label for the test_index
                predictions[test_index] = test_predictions[i]

        # get ordered predictions
        predictions = [predictions[i] for i in range(len(predictions))]
        self.write_predictions_to_file(predictions)
        self.evaluate()

if __name__ == "__main__":
    theano.config.openmp = True
    OMP_NUM_THREADS=4
    args = get_args()
    config_file = expand_path(args.config_file)
    print("Loading {}".format(config_file))
    config = yaml.load(open(config_file, "r"))
    experiment = Experiment(config)
    # plot model
    try:
        experiment.write_model_graph()
    except:
        print("Problem writing model graph")
    # run kfolds
    experiment.run_kfold()
