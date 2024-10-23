# import the necessary libraries
import json
import openai
import pandas as pd
import time

# Set your API key
api_key = ''
openai.api_key = api_key

# This is the prompt that will be used for the fine-tuning task
system_message = 'You are an expert in Science Technology Engineering and Mathematics. Classify abstracts as \"applied\" and \"basic\" research. An abstract should be applied when the research directly applies to an industrial or business application in the real-world context. Basic research is experimental, empirical or theoretical work undertaken primarily to acquire new knowledge or method or technique or test. If a new method or technique or knowledge or test is provided, it is basic. It is about understanding underlying mechanisms, phenomena and properties without immediate industrial or business application. If the abstract has a mix of applied and basic, it should be regarded as applied. Based on this principle, classify the following abstracts.'

# Function to create user message
def create_user_message(row):
    return f"""Abstract: {row['Abstract']}\n\nClass: """

# Function to prepare example conversation
def prepare_example_conversation(row):
    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": create_user_message(row)},
                {"role": "assistant", "content": row["Class"]}]
    return {"messages": messages}

# Function to write files to JSONL format
def write_jsonl(data_list, filename):
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

# The range here defines how many repetitions of the fine-tuning process you want to run. In this case, it is set to 1
for i in range(0,1):
    print('Start trial ' + str(i))
    # Before running this code, make sure you have created the training and validation sets according to your parameter needs
    training_file = f"training_{i}.xlsx"
    validation_file = f"valid_{i}.xlsx"
    test_file = f"test_set.xlsx"

    # Load datasets
    training_df = pd.read_excel(training_file)
    validation_df = pd.read_excel(validation_file)
    test_df = pd.read_excel(test_file)

    # Prepare datasets
    training_data = training_df.apply(prepare_example_conversation, axis=1).tolist()
    validation_data = validation_df.apply(prepare_example_conversation, axis=1).tolist()

    # Write data to JSONL files
    training_file_name = f"tmp_class_finetune_training_{i}.jsonl"
    validation_file_name = f"tmp_class_finetune_validation_{i}.jsonl"
    write_jsonl(training_data, training_file_name)
    write_jsonl(validation_data, validation_file_name)

    # Upload the files for fine-tuning
    training_response = openai.File.create(file=open(training_file_name, "rb"), purpose="fine-tune")
    validation_response = openai.File.create(file=open(validation_file_name, "rb"), purpose="fine-tune")

    # Create the fine-tuning job
    response = openai.FineTuningJob.create(
        training_file=training_response["id"],
        validation_file=validation_response["id"],
        model="gpt-3.5-turbo", #the model you want to use
        suffix=f"_ft_{i}"
    )
    job_id = response["id"]

    # Function to check training completion status
    def check_training_completion(job_id):
        while True:
            response = openai.FineTuningJob.list_events(id=job_id, limit=100)
            for event in reversed(response["data"]):
                print(event["message"])
                if event["message"] == 'The job has successfully completed':
                    print("Training has completed successfully.")
                    return

            time.sleep(30)  # Wait for 30 seconds before checking again

    check_training_completion(job_id)

    # Get the fine-tuned model id
    response = openai.FineTuningJob.retrieve(job_id)
    fine_tuned_model_id = response["fine_tuned_model"]
    with open(f"ft_id_{i}.txt", "w") as file:
        file.write(fine_tuned_model_id)

    # Apply the fine-tuned model on the test set
    responses = []
    for row_index in range(len(test_df)):
        test_row = test_df.loc[row_index]
        test_messages = [{"role": "system", "content": system_message},
                         {"role": "user", "content": create_user_message(test_row)}]
        response = openai.ChatCompletion.create(
            model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=500  # Set the temperature according to your needs
        )
        responses.append(response["choices"][0]["message"]["content"])
        if row_index % 200 == 0 and row_index != 0:
            print("Pausing for 1 minute...")
            time.sleep(60)  # Pause for 60 seconds

    # Create a DataFrame with the responses
    test_df['Results'] = responses
    test_df.to_excel(f"test_result_{i}.xlsx", index=False)

    # Calculate and save accuracy
    test_df['Match'] = (test_df['Results'] == test_df['Class']).astype(int)
    accuracy = test_df['Match'].mean()
    test_df['Accuracy'] = accuracy
    test_df.to_excel(f"test_result_accuracy_{i}.xlsx", index=False)
