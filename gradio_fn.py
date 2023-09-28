import gc
import torch
import os
import json
import gradio as gr


# functions used to interact with system

def clear_memory():
    # called after deleting the items in python
    gc.collect()
    torch.cuda.empty_cache()

def delete_folder(path):
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
    except:
        print("cannot delete the folder "+path)

def load_settings():
    try:
        with open("settings.json", "r") as file:
            data = json.load(file)
        return data
    except:
        print("settings failed to load")

def save_settings(info_tab_on_launch):
    try:
        with open("settings.json", "w") as file:
            data = {
                "info_tab_on_launch": info_tab_on_launch
            }
            
            json.dump(data, file, indent=4)
        return gr.JSON(value=json.dumps(data))
    except:
        print("settings failed to save")

# functions used to update gradio components (updating values or setting visibility)

# def change_tab(): # doesn't work because updating of Tabs has a bug right now
#     return gr.Tabs(selected=4)