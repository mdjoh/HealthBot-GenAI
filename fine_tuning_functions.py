# Create functions for fine-tuning

from openai import OpenAI
import os
from dotenv import load_dotenv

# ====================================== OPENAI ACCESS SETUP =======================================
# create the OpenAI object that will have the fine-tuned model attributes
# NOTE: you must have a valid OpenAI API Key stored in an .env file
def createOpenAIClient():
    # return OpenAI object with OpenAI API key loaded from environment variable stored in .env file

    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    return client

# ====================================== UPLOAD FINE-TUNING DATASET =======================================
# upload supplementary dataset for model fine-tuning to OpenAI
def uploadFineTuningData(filepath, client):
    # pass in filepath of dataset in JSONL format for LLM fine-tuning
    # return the file ID for the uploaded training data file which will be used for fine-tuning

    upload_response = client.files.create(file=open(filepath, "rb"),
                                          purpose="fine-tune")

    training_file_id = upload_response.id

    return training_file_id

# ====================================== RUN FINE-TUNING JOB =======================================
# create and launch a fine-tuning job
def createFineTuneJob(client, training_file_id, model):
    # pass in client OpenAI object, fine-tuning training data file ID, and name of LLM to fine-tune
    # return the fine-tuning job ID if the fine-tuning job is successful

    client.fine_tuning.jobs.create(training_file=training_file_id,
                                    model=model,
                                    method={"type": "supervised",
                                            "supervised": {
                                                "hyperparameters": {"n_epochs": 2}
                                            }
                                    }
                                )

# ====================================== GET FINE-TUNED MODEL ID =======================================
# function to retrieve a fine-tuned model's ID only
# serves to retrieve a fine-tuned model and avoid needlessly fine-tuning again

def getFineTunedModelID(client):
    # pass in client OpenAI object

    # create an empty list to populate job IDs
    fine_tuning_job_list = []

    # OpenAI platform populates fine-tuning jobs in reverse chronological order
    for job in client.fine_tuning.jobs.list():
        fine_tuning_job_list.append(job.id)

    # retrieve the ID of the most recent fine-tuning job which has the model to deploy
    fine_tuning_job_id = client.fine_tuning.jobs.retrieve(fine_tuning_job_list[0])

    # return the fine-tuned model ID from the job of interest
    # the fine-tuned model is the model to deploy
    fine_tuned_model_id = fine_tuning_job_id.fine_tuned_model
    return fine_tuned_model_id
