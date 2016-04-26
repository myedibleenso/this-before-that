import pandas as pd

def compute_performance(results_file):
    """
    Takes a tab-delimited file of results (Gold, Predicted)
    and produces a dataframe of performance
    """
    df = pd.read_csv(results_file, delimiter="\t")
    smoothing_value = 0.00001

    gold = df.Gold.values
    predictions = df.Predicted.values
    class_labels = sorted(set(df.Gold.values))
    def performance(df, label_of_interest):
        tp = len(df[(df.Gold == label_of_interest) & (df.Predicted == label_of_interest)])
        tn = len(df[(df.Gold != label_of_interest) & (df.Predicted != label_of_interest)])
        fp = len(df[(df.Gold != label_of_interest) & (df.Predicted == label_of_interest)])
        fn = len(df[(df.Gold == label_of_interest) & (df.Predicted != label_of_interest)])
        support = len(df[df.Gold == label_of_interest])
        p = tp / (tp + fp + smoothing_value)
        r = tp / (tp + fn + smoothing_value)
        f1 = 2 * ((p * r) / (p + r + smoothing_value))
        return [label_of_interest, p, r, f1, support]
    header = ["Class", "Precision", "Recall", "F1", "Support"]
    rows = [performance(df, label) for label in class_labels]
    macro_p = 0
    macro_r = 0
    macro_f1 = 0
    support_total = 0
    for row in rows:
        macro_p += row[1]
        macro_r += row[2]
        macro_f1 += row[3]
        support_total += row[-1]
    macro_p /= len(rows)
    macro_r /= len(rows)
    macro_f1 /= len(rows)
        
    rows.append(["TOTAL (macro)", macro_p, macro_r, macro_f1, support_total])

    return pd.DataFrame(rows, columns=header)



def compute_performance_from_df(df):
    """
    Takes a Pandas.DataFrame with Gold and Predicted columns
    and produces a Dataframe of performance
    """
    smoothing_value = 0.00001

    gold = df.Gold.values
    predictions = df.Predicted.values
    class_labels = sorted(set(df.Gold.values))
    def performance(df, label_of_interest):
        tp = len(df[(df.Gold == label_of_interest) & (df.Predicted == label_of_interest)])
        tn = len(df[(df.Gold != label_of_interest) & (df.Predicted != label_of_interest)])
        fp = len(df[(df.Gold != label_of_interest) & (df.Predicted == label_of_interest)])
        fn = len(df[(df.Gold == label_of_interest) & (df.Predicted != label_of_interest)])
        support = len(df[df.Gold == label_of_interest])
        p = tp / (tp + fp + smoothing_value)
        r = tp / (tp + fn + smoothing_value)
        f1 = 2 * ((p * r) / (p + r + smoothing_value))
        return [label_of_interest, p, r, f1, support]
    header = ["Class", "Precision", "Recall", "F1", "Support"]
    rows = [performance(df, label) for label in class_labels]
    macro_p = 0
    macro_r = 0
    macro_f1 = 0
    support_total = 0
    for row in rows:
        macro_p += row[1]
        macro_r += row[2]
        macro_f1 += row[3]
        support_total += row[-1]
    macro_p /= len(rows)
    macro_r /= len(rows)
    macro_f1 /= len(rows)
        
    rows.append(["TOTAL (macro)", macro_p, macro_r, macro_f1, support_total])

    return pd.DataFrame(rows, columns=header)
#performance = compute_performance("results.tsv")

#performance.round(2)