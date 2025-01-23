# Make supplementary dataset for GPT model fine-tuning
# Convert csv dataset into JSONL format containing question-answer pairs for
# assistant (disease) and user (symptoms) chat

# ======================= DATA SETUP ================================
import pandas as pd
import json
import random

# read in disease & symptom dataset
data_df = pd.read_csv('./DiseaseAndSymptoms.csv', header=0)

# convert disease & symptom data into string format
data_df = data_df.astype(str)

# put column names of disease & symptom dataset in a list
symptom_col_names = list(data_df.columns)
symptom_col_names.remove('Disease') # remove 'Disease' element from column name list to only have symptom column names remaining

# ======================= DATASET PREPROCESSING ================================
# make text preprocessing functions
# function to remove space character from symptom
def removeSpace(symptom):
    return symptom.replace(' ', '')

# function to remove space character from symptom
def replaceUnderscore(symptom):
    return symptom.replace('_', ' ')

# apply text preprocessing functions to disease & symptom dataframe

# make all data lowercase
data_df = data_df.apply(lambda x: x.str.lower())

# remove space and replace underscore character with a space character for symptom features
for col_name in symptom_col_names:
    data_df[col_name] = data_df[col_name].apply(removeSpace)
    data_df[col_name] = data_df[col_name].apply(replaceUnderscore)

# ======================= PARSE DATASET ================================
# make disease into its own list for assistant element of conversation
disease_list = data_df['Disease'].values.tolist()

# drop disease column from data_df to leave with only symptom features
symptom_df = data_df.drop(columns=['Disease'])

# ==================== AGGREGATE USER CHAT MESSAGES ==========================
# function to concatenate all symptom strings separated by ', '
# take in a dataframe row series of symptoms which are each a string
# take the row series values and make a list of symptoms without nan
# convert the produced list to a string via ', '.join(list)
def concatenateSymptoms(row_series):

    # put the symptoms from the row series into a list but remove the nan
    concatenated_symptoms_list = [element for element in list(row_series.values) if element != 'nan']

    # convert list of symptoms into a string separated by ', '
    concatenated_symptoms = ', '.join(concatenated_symptoms_list)

    return concatenated_symptoms

# set up empty list to append symptom strings to
training_data_symptoms = []

# get the number of rows in disease & symptom dataset
num_rows = len(disease_list)

# iterate over data_df rows and call concatenate_symptoms function in each iteration
for row in range(num_rows):
    symptom_string = concatenateSymptoms(data_df.iloc[row])

    # append return string to the user symptom list
    training_data_symptoms.append(symptom_string)

# concatenate list of diseases and training data symptoms to generate a symptoms-disease tuple list
training_data = list(zip(training_data_symptoms, disease_list))

# ============================= WRITE CHAT TO A JSONL FILE =================================
# randomly select 20 samples from symptoms-disease tuple list for use in LLM fine-tuning
# only using 20 samples for fine-tuning to limit cost; if desired, can use more training data samples
number_of_samples = 20
data_to_write = random.sample(training_data, number_of_samples)

# write the symptoms-disease pairs to the JSONL output file in write mode
with open('./chat_data_for_fine_tuning.jsonl', 'w') as file:

    symptoms_col_num = 0
    disease_col_num = 1

    # iterate over each row of disease & symptom dataset and write each question-answer pair of a
    # user-assistant chat to the output JSONL file
    for i in range(number_of_samples):
        chat_format = {"messages": [
                    {"role": "developer", "content": "You are a medical professional. Try to give the most likely diagnosis based on given symptoms."},
                    {"role": "user", "content": "What could I have if I am experiencing " + data_to_write[i][symptoms_col_num] + "?"},
                    {"role": "assistant", "content": data_to_write[i][disease_col_num]}
                ]}

        # write formatted chat to JSONL file
        if i == number_of_samples - 1:
            file.write(json.dumps(chat_format)) # can't have a new line at the end in output file for fine-tuning
        else:
            file.write(json.dumps(chat_format) + '\n')
