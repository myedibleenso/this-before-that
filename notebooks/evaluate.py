import pandas as pd

class Score(object):
    def __init__(self, class_label, p, r, f1, tp, fp, fn):
        self.class_label = class_label
        self.precision = p
        self.recall = r
        self.f1 = f1
        self.tp = tp
        self.fp = fp
        self.fn = fn
        
    def summary(self):
        print("{}\tPrecision\tRecall\tF1".format(self.class_label))
        print("{:.2f}\t{:.2f}\t{:.2f}".format(self.precision, self.recall, self.f1))

class Evaluator(object):
    def __init__(self, results_file, neg_class = "None"):
        self.results_file = results_file
        self.neg_class = neg_class
        self.smoothing = 0.00001
        
    def compute_class_score(self, class_label):
        df = pd.read_csv(self.results_file, delimiter="\t")
        tp = len(df[(df.Gold == df.Predicted) & (df.Gold == class_label)])        
        fp = len(df[(df.Gold != df.Predicted) & (df.Predicted == class_label) & (df.Gold != class_label)])
        p = tp / (tp + fp + self.smoothing)
        fn = len(df[(df.Gold == class_label)]) - tp
        #print("Total annos for {}: {}".format(class_label, tp + fn))
        r = tp / (tp + fn + self.smoothing)
        f1 = (2 * p * r) / (p + r + self.smoothing)
        return Score(class_label, p, r, f1, tp, fp, fn)
    
    def compute_micro_score(self):
        df = pd.read_csv(self.results_file, delimiter="\t")
        # ignore instances where negative class was predicted
        pos_predictions = len(df[df.Predicted != self.neg_class])
        # num. positive predictions that are correct
        tp = len(df[(df.Gold == df.Predicted) & (df.Predicted != self.neg_class)])
        # num. positive predictions that are incorrect
        fp = len(df[(df.Gold != df.Predicted) & (df.Predicted != self.neg_class)])
        fn = len(df[df.Gold != self.neg_class]) - tp
        p = tp / (tp + fp + self.smoothing)
        r = tp / (tp + fn + self.smoothing)
        f1 = (2 * p * r) / (p + r + self.smoothing)

        return Score("MICRO", p, r, f1, tp, fp, fn)
        
    def compute_macro_score(self):
        df = pd.read_csv(self.results_file, delimiter="\t")
        scores = [self.compute_class_score(cl) for cl in set(df.Gold.values)]
        p = 0
        r = 0
        f1 = 0
        tp = 0
        fp = 0
        fn = 0
        pos_classes = [s for s in scores if s.class_label != self.neg_class]
        for s in pos_classes:    
            p += s.precision
            r += s.recall
            f1 += s.f1
            tp += s.tp
            fp += s.fp
            fn += s.fn
            
        return Score("MACRO", p/len(pos_classes), r/len(pos_classes), f1/len(pos_classes), tp, fp, fn)
            
    def generate_scores_df(self):
        df = pd.read_csv(self.results_file, delimiter="\t")
        header = ("Class", "P", "R", "F1", "TP", "FP", "FN")
        scores = [self.compute_class_score(cl) for cl in sorted(set(df.Gold.values))]
        data = [(s.class_label, s.precision, s.recall, s.f1, s.tp, s.fp, s.fn) for s in scores]
        macro = self.compute_macro_score()
        micro = self.compute_micro_score()
        data += [(macro.class_label, macro.precision, macro.recall, macro.f1, macro.tp, macro.fp, macro.fn), (micro.class_label, micro.precision, micro.recall, micro.f1, micro.tp, micro.fp, micro.fn)]
        return pd.DataFrame(data, columns=header)