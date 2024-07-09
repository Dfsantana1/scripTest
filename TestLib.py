from robin_api import RobinAIClient

# Inicializar clientes
api_key="Q9JccNBNBZqZEfGp7iEPt95E89RFJY"
client1 = RobinAIClient(api_key=api_key)
client2 = RobinAIClient(api_key=api_key)
client3 = RobinAIClient(api_key=api_key)
client4 = RobinAIClient(api_key=api_key)
client5 = RobinAIClient(api_key=api_key)
client6 = RobinAIClient(api_key=api_key)
client7 = RobinAIClient(api_key=api_key)
client8 = RobinAIClient(api_key=api_key)
client9= RobinAIClient(api_key=api_key)
client10= RobinAIClient(api_key=api_key)
client11= RobinAIClient(api_key=api_key)

print("hi started the test by RobinAI API")

# Booleanos para seguimiento de éxito
text_to_image_success = False
create_stream_success = False
create_response_success = False
create_folder_success = False
get_similar_sentences_success = False
get_response_similar_sentences_stream_success = False
get_response_similar_sentences_success = False
get_folder_files_success = False
upload_data_set_success = False
start_finetuning_success = False
fine_tuning_media_id = False
fine_tuning_detail=False
fine_tuning_list=False



# 1. Generar imagen a partir de texto
print("1. Generate to image")
try:
    image = client1.completions.text_to_image(prompt="A beautiful sunset over the ocean with a pink and purple sky.")
    print("\nThe URL is:", image.url)
    text_to_image_success = True
except AttributeError as e:
    print(f"Function text_to_image failed: {e}")

# 2. Crear stream
print("2. Create stream")
conversation = [
    {"role": "system", "content": "system_prompt"},
    {"role": "user", "content": "que es una llama ?"}
]
try:
    stream = client2.completions.create_stream(
        model="ROBIN_4",
        conversation=conversation,
        max_tokens=200,
        save_response=False,
        temperature=1
    )
    for chunk in stream:
        if not chunk.choices[0].finish_reason:
            print(chunk.choices[0].delta.content, end="")
        else:
            print(chunk.details, end="")
    create_stream_success = True
except AttributeError as e:
    print(f"Function create_stream failed: {e}")

# 3. Obtener respuesta similar a oraciones
print("3. Get response similar sentences stream")
try:
    get_answer = client3.completions.create(
        model="ROBIN_4",
        conversation=conversation,
        max_tokens=200,
        save_response=False,
        temperature=1
    )
    print(get_answer, end="")
    create_response_success = True
except AttributeError as e:
    print(f"Function create_response failed: {e}")

# 4. Crear carpeta y subir archivo
print("4. Create Folder")
try:
    folder_information = client4.files.upload_file(url="https://arxiv.org/pdf/2302.13971.pdf")
    apiFolderId = folder_information.folder.apiFolderId
    print(folder_information.folder.apiFolderId)
    create_folder_success = True
except AttributeError as e:
    print(f"Function create_folder failed: {e}")

# 5. Obtener oraciones similares
print("5. Get similar sentences")
try:
    files = client5.files.get_similar_sentences(
        query="What are the practical implications of the findings in the document?",
        top=15,
        api_folder_id=apiFolderId,
        similarity_threshold=0.4
    )
    print(files)
    get_similar_sentences_success = True

    files_get = client6.files.get_response_similar_sentences_stream(
        model="ROBIN_4",
        max_new_tokens=200,
        top=1,
        api_folder_id=apiFolderId,
        similarity_threshold=0.4,
        conversation=conversation,
        only_with_context=True
    )
    for chunk in files_get:
        if not chunk.choices[0].finish_reason:
            print(chunk.choices[0].delta.content, end="")
        else:
            print(chunk.details, end="")
    get_response_similar_sentences_stream_success = True

    file_test = client7.files.get_response_similar_sentences(
        model="ROBIN_4",
        max_new_tokens=200,
        top=1,
        api_folder_id=apiFolderId,
        similarity_threshold=0.4,
        conversation=conversation,
        only_with_context=True
    )
    print(file_test.message.choices[0].message.content)
    get_response_similar_sentences_success = True

    get_folder_files = client8.files.get_folder_files(api_folder_id=apiFolderId)
    print(get_folder_files)
    get_folder_files_success = True
except AttributeError as e:
    print(f"Function get_similar_sentences or subsequent functions failed: {e}")



print('6. upload data set ');
try:
    file_path="./dataset.csv"
    purpose = "propósito_de_subida"
    description = "descripción_de_tu_archivo"

    finetunings=client9.fine_tuning.upload_local_file( file=file_path, purpose=purpose, description=description)
    print(finetunings)
    fine_tuning_media_id = finetunings.id
    upload_data_set_success = True
except AttributeError as e:
    print(f"Function get_similar_sentences or subsequent functions failed: {e}")


# 7. Iniciar ajuste fino
print("7. Start fine-tuning")

try:
    fineTuning = client11.fine_tuning.star_fine_tuning(
        model="Robin-7B-Instruct",
        task="language-modeling",
        sub_category="chat",
        media_id=fine_tuning_media_id,
        description="description",
        params={
    "optim": "adamw_8bit",
    "learning_rate": 0.0001,
    "max_grad_norm": 0.3,
    "num_train_epochs": 3,
    "evaluation_strategy": "epoch",
    "eval_steps": 7000,
    "warmup_ratio": 0.05,
    "save_strategy": "epoch",
    "group_by_length": True,
    "lr_scheduler_type": "cosine"
    },
        params_input={
    "input_template": "### GIVEN THE CONTEXT: {context} ### INSTRUCTION: {question} ### RESPONSE IS: "
    },
        params_output='answer',
        extension="csv",
    )
    print(fineTuning)
    start_finetuning_success = True
except AttributeError as e:
    print(f"Function start_finetuning failed: {e}")

if(start_finetuning_success):
    print('8. get fine tuning status');
    try:
        fineTuningStatus = client10.fine_tuning.get_fine_tuning_detail(fine_tuning_id= fineTuning.FineTuningid)
        print(fineTuningStatus)
        fine_tuning_detail = True
    except AttributeError as e:
        print(f"Function get_fine_tuning_status failed: {e}")

# print('9. get fine tuning list');
# try:
#     fineTuningList = client11.fine_tuning.get_fine_tuning_results()
#     print(fineTuningList)
#     fine_tuning_list = True
# except AttributeError as e:
#     print(f"Function get_fine_tuning_list failed: {e}")

        

# Resultados finales
print(f"\ntext_to_image completed: {text_to_image_success}")
print(f"create_stream completed: {create_stream_success}")
print(f"create_response completed: {create_response_success}")
print(f"create_folder completed: {create_folder_success}")
print(f"get_similar_sentences completed: {get_similar_sentences_success}")
print(f"get_response_similar_sentences_stream completed: {get_response_similar_sentences_stream_success}")
print(f"get_response_similar_sentences completed: {get_response_similar_sentences_success}")
print(f"get_folder_files completed: {get_folder_files_success}")
print(f"upload_data_set completed: {upload_data_set_success}")
print(f"start_finetuning completed: {start_finetuning_success}")
print(f"fine_tuning_detail completed: {fine_tuning_detail}")
print(f"fine_tuning_list completed: {fine_tuning_list}")

