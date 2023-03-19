import pandas as pd
import random

def generate_synaptic_weights():
    return [round(random.uniform(0, 1), 4) for _ in range(4)]

def calculate_sample(synaptic_weights, values):
    u = sum(weight * value for weight, value in zip(synaptic_weights, values))
    return 1 if (u > 0) else -1

def update_weights(weights, learning_rate, obtained, row):
    return [old_weight + learning_rate * ((row['d'] - obtained) * row[f'x{index}']) for index, old_weight in enumerate(weights)]

def is_wanted_equals_obtained(wanted, obtained):
    if (wanted > 0 and obtained > 0) or (wanted < 0 and obtained < 0):
        return True
    else:
        return False

def run_tests(synaptic_weights, df):
    predicted_values = []
    for index, row in df.iterrows():
        sample = calculate_sample(synaptic_weights, row[:-1])
        predicted_values.append(sample)
    df['d_predicted'] = predicted_values
    return df

def train_model(data, learning_rate, interactions_amount_limit):
    interactions_amount = 0
    has_converged = False
    synaptic_weights = generate_synaptic_weights()
    print(f"Pesos iniciais: {synaptic_weights}")
    
    while not (has_converged or (interactions_amount >= interactions_amount_limit)):
        num_errors = 0

        for index, row in data.iterrows():
            sample = calculate_sample(synaptic_weights, row)
            
            if not is_wanted_equals_obtained(row['d'], sample):
                num_errors +=1
                synaptic_weights = update_weights(synaptic_weights, learning_rate, sample, row)

        interactions_amount += 1
        has_converged = (num_errors == 0)
    
    print(f"Epocas: {interactions_amount}\nErros: {num_errors}\nPesos Finais: {synaptic_weights}")
    return synaptic_weights

def run(training_df, test_df, learning_rate, interactions_amount_limit):
    conveged_synaptic_weights = train_model(training_df, learning_rate, interactions_amount_limit)
    print(run_tests(conveged_synaptic_weights, test_df))
