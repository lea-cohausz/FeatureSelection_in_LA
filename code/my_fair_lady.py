import subprocess
from utils import correlation_remover2
import pandas as pd

def cade(X_train, X_test, sensitive_columns, dominant_data_type = "discrete"):
    mb_target, desc_sensitive = learn_mb_descendants(dominant_data_type)
    mb_target = [i for i in mb_target if i not in sensitive_columns]
    desc_and_mb = list(set(mb_target) & set(desc_sensitive))
    if desc_and_mb:
        desc_and_mb = desc_and_mb + sensitive_columns
        X_train_decorrelate = X_train[desc_and_mb]
        X_test_decorrelate = X_test[desc_and_mb]
        X_train_decorrelated, X_test_decorrelated = correlation_remover2(X_train_decorrelate, X_test_decorrelate, sensitive_columns, alpha=1.0)
        X_train = X_train.drop(desc_and_mb, axis=1)
        X_test = X_test.drop(desc_and_mb, axis=1)
        X_train = pd.concat([X_train, X_train_decorrelated], axis=1)
        X_test = pd.concat([X_test, X_test_decorrelated], axis=1)
        X_train = X_train[mb_target]
        X_test = X_test[mb_target]
    elif mb_target:
        X_train = X_train[mb_target]
        X_test = X_test[mb_target]
    else:
        X_train = X_train.drop(sensitive_columns, axis=1)
        X_test = X_test.drop(sensitive_columns, axis=1)
    return X_train, X_test


def learn_mb_descendants(dominant_data_type = "discrete"):
    if dominant_data_type == "discrete":
        result_mb = subprocess.run(
            ['Rscript', 'bnlearn_hc_disc.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )
        output_mb = result_mb.stdout.strip()  # Remove any trailing newline or spaces
        mb_target = output_mb.split()

        result_desc = subprocess.run(
            ['Rscript', 'bnlearn_desc_disc.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )
        output_desc = result_desc.stdout.strip()  # Remove any trailing newline or spaces
        desc_sensitive = output_desc.split()

    else:
        result_mb = subprocess.run(
            ['Rscript', 'bnlearn_hc_cont.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )
        output_mb = result_mb.stdout.strip()  # Remove any trailing newline or spaces
        mb_target = output_mb.split()

        result_desc = subprocess.run(
            ['Rscript', 'bnlearn_desc_cont.R'],  # Make sure 'Rscript' is in your PATH
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Capture output as string
        )
        output_desc = result_desc.stdout.strip()  # Remove any trailing newline or spaces
        desc_sensitive = output_desc.split()
    return mb_target, desc_sensitive