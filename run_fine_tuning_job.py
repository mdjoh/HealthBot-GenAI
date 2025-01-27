# Create and run a GPT fine-tuning job

from fine_tuning_functions import createOpenAIClient, uploadFineTuningData, createFineTuneJob

# create OpenAI client
client = createOpenAIClient()

# upload supplementary dataset for model fine-tuning to OpenAI platform
training_file_id = uploadFineTuningData('./fine_tuning_data/df_chat_data_for_fine_tuning.jsonl', client)

# create and launch fine-tuning job
fine_tuning_job_id = createFineTuneJob(client, training_file_id, model="gpt-4o-mini-2024-07-18")
