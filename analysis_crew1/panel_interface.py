


    # panel_interface.py

import panel as pn
import threading
import time
from pathlib import Path

pn.extension(design="material")

uploaded_files = []
uploaded_filenames = []

message_params = dict(
    default_avatars={"User": "👨🏾‍🦱"},
)

avatars = {
    "Summarizer": "man.png",
    "Query_Agent": "man.png",
    "Analysis_Agent": "man.png"
}

user_input = None
initiate_chat_task_created = False
start_crew_callback = None

def initiate_chat(message):
    global initiate_chat_task_created
    initiate_chat_task_created = True
    if start_crew_callback:
        start_crew_callback(message)
def callback(contents: str, user: str, instance: pn.chat.ChatInterface):

    global initiate_chat_task_created
    global user_input

    if not initiate_chat_task_created:
        thread = threading.Thread(target=initiate_chat, args=(contents,))
        thread.start()
    else:
        user_input = contents



def handle_file_upload(event):
    global uploaded_files, uploaded_filenames
    new_files = file_input.value
    new_filenames = file_input.filename
    
    if new_files:
        if isinstance(new_files, list):
            uploaded_files.extend(new_files)
        else:
            uploaded_files.append(new_files)
        
        if isinstance(new_filenames, list):
            uploaded_filenames.extend(new_filenames)
        else:
            uploaded_filenames.append(new_filenames)
        
        if len(uploaded_files) == 1:
            chat_interface.send(f"Added {len(uploaded_files)} file to the upload queue.", user="System", respond=False)
        else:
            chat_interface.send(f"Added {len(uploaded_files)} file(s) to the upload queue.", user="System", respond=False)

def process_files(event):
    global uploaded_files, uploaded_filenames
    save_folder_path = "C:/Users/omarmorris/Desktop/analysis_crew1/documents"
    
    
    for file_content, file_name in zip(uploaded_files, uploaded_filenames):
        save_path = Path(save_folder_path, file_name)
        with open(save_path, mode='wb') as w:
            w.write(file_content)
        if save_path.exists():
            chat_interface.send(f"File '{file_name}' uploaded to directory successfully!", user="System", respond=False)
    
    uploaded_files.clear()
    uploaded_filenames.clear()

file_input = pn.widgets.FileInput(name="Upload Documents")
file_input.param.watch(handle_file_upload, 'value')

process_button = pn.widgets.Button(name="Upload Files", button_type="primary")
process_button.on_click(process_files)
chat_interface = pn.chat.ChatInterface(message_params=message_params, callback=callback)
chat_interface.send("Please upload your documents. We accept the following formats: pdf, txt and docx. Choose your files, click the Upload files button and enter your query", user="System", respond=False)

template = pn.template.MaterialTemplate(title='Document Analysis App')

template.sidebar.append(pn.pane.Markdown("# Upload and Process Files"))
template.sidebar.append(file_input)
template.sidebar.append(process_button)

template.main.append(chat_interface)



MAX_SIZE_MB = 150

def serve_app():
    pn.serve(
        template,
        websocket_max_message_size=MAX_SIZE_MB * 1024 * 1024,
        http_server_kwargs={'max_buffer_size': MAX_SIZE_MB * 1024 * 1024}
    )

def register_start_crew_callback(callback):
    global start_crew_callback
    start_crew_callback = callback
