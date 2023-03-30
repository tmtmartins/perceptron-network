import pandas as pd
from perceptron import run
import sys

data_file = "./Dados.ods"
training_df = pd.read_excel(data_file, sheet_name="Treinamento")
test_df = pd.read_excel(data_file, sheet_name="Teste")
learning_rate = 0.01
interactions_amount_limit = 500

if __name__ == "__main__":
    with open('output.txt', 'a') as f:
        sys.stdout = f
        print("\n\n5:")
        run(training_df, test_df, learning_rate, interactions_amount_limit)
