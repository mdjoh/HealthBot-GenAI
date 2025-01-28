# Script to make supplementary dataset for GPT model fine-tuning
# Convert csv dataset into JSONL format containing question-answer pairs for
# assistant (disease) and user (symptoms) chat

# =============================== DATA SETUP ===================================
import pandas as pd
import json

# read in disease & symptom dataset
data_df = pd.read_csv('./DiseaseAndSymptoms.csv', header=0)

# convert disease & symptom data into string format
data_df = data_df.astype(str)

# parse data_df into disease and symptom dataframe
disease_df = data_df['Disease']
symptom_df = data_df.drop(columns=['Disease'])

# ======================= DATASET PREPROCESSING ================================
# make text preprocessing functions

# DISEASE:
# function to make disease names lowercase except for hepatitis type and GERD
def processDiseaseData(disease):

    # remove leading and training spaces in disease name
    disease = disease.strip()

    if disease.lower().startswith("hepatitis"):
        disease = "hepatitis " + disease[-1].upper() # last character in a disease string that starts with 'hepatitis' is the type; ensure type is uppercase
    elif disease != "GERD" and disease != "AIDS": # don't make 'GERD' and 'AIDS' lowercase
        disease = disease.lower()

    return disease

# apply disease data preprocessing function
disease_df = disease_df.apply(processDiseaseData)

# SYMPTOM:
# function to make symptom data lowercase and remove leading and trailing spaces and underscore character from a symptom string in symptom dataframe
def processSymptomData(symptom):
    symptom = symptom.strip()
    symptom = symptom.lower()
    symptom = symptom.replace('_', ' ')

    return symptom

# put symptom column names in a list
symptom_col_names = list(symptom_df.columns)

# apply symptom data preprocessing function to all symptom columns
for col_name in symptom_col_names:
    symptom_df[col_name] = symptom_df[col_name].apply(processSymptomData)

# ========================== AGGREGATE USER CHAT MESSAGES ================================
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

# get the number of rows in dataset which will be the number of chat entries in training data
num_rows = len(symptom_df)

# make disease into its own list for assistant element of conversation
disease_list = disease_df.values.tolist()

# set up empty list to append symptom strings to
training_data_symptoms = []

# iterate over data_df rows and call concatenate_symptoms function in each iteration
for row in range(num_rows):
    symptom_string = concatenateSymptoms(symptom_df.iloc[row])

    # append return string to the user symptom list
    training_data_symptoms.append(symptom_string)

# concatenate list of diseases and training data symptoms to generate a symptoms-disease tuple list
training_data = list(zip(training_data_symptoms, disease_list))

# ================================== WRITE CHAT TO A JSONL FILE ======================================
# write the symptoms-disease pairs to the JSONL output file in write mode
with open('./fine_tuning_data/chat_data_for_fine_tuning.jsonl', 'w') as file:

    symptoms_col_num = 0
    disease_col_num = 1

    # iterate over each row of disease & symptom dataset and write each question-answer pair of a
    # user-assistant chat to the output JSONL file
    for i in range(num_rows):
        chat_format = {"messages": [
                    {"role": "system", "content": "You are a medical professional. Try to give the most likely diagnosis based on the symptoms entered by the user."},
                    {"role": "user", "content": "What could I have if I am experiencing " + training_data[i][symptoms_col_num] + "?"},
                    {"role": "assistant", "content": training_data[i][disease_col_num]}
                ]}

        # write formatted chat to JSONL file
        if i == num_rows - 1:
            file.write(json.dumps(chat_format)) # can't have a new line at the end in output file for fine-tuning
        else:
            file.write(json.dumps(chat_format) + '\n')
