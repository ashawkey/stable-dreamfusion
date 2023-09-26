import os
import gradio as gr
import gradio_fn
import torch

def delete_directory(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

if __name__ == '__main__':
    with gr.Blocks(title="zero123d reconstruction") as demo:
        gr.Markdown(
            """
            # Image to 3D Model Generation
            FYP by Rod Oh Zhi Hua.
            """)
        
        with gr.Tab("show generated models"):
            max_epoch = gr.Number(value=50, visible=False, precision=0) #50
            dmtet = gr.Checkbox(value=False, visible=False)
            with gr.Row():
                image_output = gr.Image(height=512, width=512, interactive=False)
            image_slider_input = gr.Slider(label="image (slide to change angle)", minimum=0, maximum=0, step=1)
            workspaces = os.listdir("workspaces/")
            workspace_input = gr.Dropdown(label="workspace name", choices=workspaces)
            load_button = gr.Button(label="load")
            file_output = gr.File(visible=False)
            # events
            image_slider_input.change(fn=gradio_fn.update_image_slider, inputs=[max_epoch, workspace_input, image_slider_input, dmtet], outputs=[image_output])
            workspace_input.focus(fn=gradio_fn.update_workspaces_list, outputs=[workspace_input])
            load_button.click(fn=gradio_fn.load_3d_model, inputs=[max_epoch, workspace_input], outputs=[max_epoch, image_output, image_slider_input, file_output])

        # with gr.Tab("preprocess image"):
        #     with gr.Row():
        #         image_input = gr.Image()
        #         rgba_output = gr.Image()
        #     with gr.Row():
        #         depth_output = gr.Image()
        #         normal_output = gr.Image()
        #     generate_button = gr.Button(value="process")
        #     generate_button.click(fn=gradio_fn.preprocess_images, inputs=image_input, outputs=[rgba_output, depth_output, normal_output])
            
        with gr.Tab("generate novel views"):
            with gr.Row():
                image_input = gr.Image(height=512, width=512)
                image_output = gr.Image(height=512, width=512)
            image_slider = gr.Slider(label="image (slide to change angle)", minimum=1, maximum=1, step=1)
            polar_input = gr.Number(label="polar", value=0.0) # 90.0
            azimuth_input = gr.Number(label="azimuth", value=30.0) # 0.0
            radius_input = gr.Number(label="radius", value=0.0) # 3.2
            generate_button = gr.Button(value="generate")
            generate_button.click(fn=gradio_fn.generate_novel_view, inputs=[image_input, polar_input, azimuth_input, radius_input], outputs=image_output)
            
        with gr.Tab("generate novel views (radius discovery)"):
            with gr.Row():
                image_input = gr.Image(height=512, width=512)
                image_output1 = gr.Image(height=512, width=512)
                image_output2 = gr.Image(height=512, width=512)
            with gr.Row():
                image_output3 = gr.Image(height=512, width=512)
                image_output4 = gr.Image(height=512, width=512)
                image_output5 = gr.Image(height=512, width=512)
            with gr.Row():
                image_output6 = gr.Image(height=512, width=512)
                image_output7 = gr.Image(height=512, width=512)
                image_output8 = gr.Image(height=512, width=512)
            with gr.Row():
                image_output9 = gr.Image(height=512, width=512)
                image_output10 = gr.Image(height=512, width=512)
                image_output11 = gr.Image(height=512, width=512)
            image_slider = gr.Slider(label="image (slide to change angle)", minimum=1, maximum=1, step=1)
            polar_input = gr.Number(label="polar", value=0.0) # 90.0
            azimuth_input = gr.Number(label="azimuth", value=30.0) # 0.0
            generate_button = gr.Button(value="generate")
            generate_button.click(fn=gradio_fn.generate_novel_view_radii, 
                                  inputs=[image_input, polar_input, azimuth_input], 
                                  outputs=[image_output1, image_output2, image_output3, image_output4, image_output5, image_output6, image_output7, image_output8, image_output9, image_output10, image_output11])
            
        # with gr.Tab("generate 3d model"):
        #     with gr.Row():
        #         image_input = gr.Image(height=512, width=512)
        #         image_output = gr.Image(height=512, width=512)
        #     image_slider_input = gr.Slider(label="image (slide to change angle)", minimum=1, maximum=1, step=1)
        #     with gr.Row():
        #         workspace_input = gr.Textbox(label="workspace name", value="test") #trial_image
        #         seed_input = gr.Number(label="seed", precision=0, value=None)
        #     size_input = gr.Number(label="size (n^2, 64 really recommended.)", value=64, precision=0, step=1) #64
        #     with gr.Row():
        #         iters_input = gr.Number(label="iters (iterations)", value=5000, precision=0, step=1) #5000
        #         lr_input = gr.Number(label="lr (learning rate)", value=1e-3) #1e-3
        #         batch_size_input = gr.Number(label="batch_size", value=1, precision=0, step=1) #1
        #     with gr.Row():
        #         dataset_size_train_input = gr.Number(label="dataset_size_train", value=100, precision=0, step=1) #100
        #         dataset_size_valid_input = gr.Number(label="dataset_size_valid", value=8, precision=0, step=1) #8
        #         dataset_size_test_input = gr.Number(label="dataset_size_test", value=100, precision=0, step=1) #100
        #     with gr.Accordion(label="advanced", open=False):
        #         backbone_input = gr.Dropdown(label="backbone (NeRFNetwork)", choices=["grid", "vanilla", "grid_tcnn", "grid_taichi"], value="grid") #nVidia
        #         optim_input = gr.Dropdown(label="optim (optimizer)", choices=["adan", "adam"], value="adan")
        #         fp16_input = gr.Checkbox(label="fp16 (use float16 for training)", value=True)
        #         max_epoch = gr.Number(value=50, visible=False, precision=0) #50
        #         dmtet = gr.Checkbox(value=False, visible=False)
        #     generate_button = gr.Button(value="generate")
        #     file_output = gr.File(visible=False)
        #     # events
        #     image_slider_input.change(fn=gradio_fn.update_image_slider, inputs=[max_epoch, workspace_input, image_slider_input, dmtet], outputs=[image_output])
        #     iters_input.input(fn=gradio_fn.update_max_epoch_calculate, inputs=[iters_input, dataset_size_train_input, batch_size_input], outputs=[max_epoch])
        #     dataset_size_train_input.input(fn=gradio_fn.update_max_epoch_calculate, inputs=[iters_input, dataset_size_train_input, batch_size_input], outputs=[max_epoch])
        #     batch_size_input.input(fn=gradio_fn.update_max_epoch_calculate, inputs=[iters_input, dataset_size_train_input, batch_size_input], outputs=[max_epoch])
        #     generate_button.click(fn=gradio_fn.generate_3d_model, 
        #                           inputs=[image_input, workspace_input, seed_input, size_input, iters_input, lr_input, batch_size_input, dataset_size_train_input, 
        #                                   dataset_size_valid_input, dataset_size_test_input, backbone_input, optim_input, fp16_input, max_epoch], 
        #                           outputs=[image_slider_input, file_output]) # the output is to update the maximum value of the image_slider which will trigger the "update_image_slider" event to load the image

        with gr.Tab("dmtet finetuning 3d model"):
            with gr.Row():
                image_output = gr.Image(height=512, width=512)
            image_slider_input = gr.Slider(label="image (slide to change angle)", minimum=1, maximum=1, step=1)
            with gr.Row():
                workspace_input = gr.Dropdown(label="workspace name", choices=workspaces)
                seed_input = gr.Number(label="seed", precision=0, value=None)
            tet_grid_size_input = gr.Dropdown(label="tet_grid_size", choices=["32", "64", "128", "256"], value="128")
            with gr.Row():
                iters_input = gr.Number(label="iters (iterations)", value=10000, precision=0, step=1) #5000
                lr_input = gr.Number(label="lr (learning rate)", value=1e-3) #1e-3
            with gr.Accordion(label="advanced", open=False):
                backbone_input = gr.Dropdown(label="backbone (NeRFNetwork)", choices=["grid", "vanilla", "grid_tcnn", "grid_taichi"], value="grid") #nVidia
                optim_input = gr.Dropdown(label="optim (optimizer)", choices=["adan", "adam"], value="adan")
                fp16_input = gr.Checkbox(label="fp16 (use float16 for training)", value=True)
                max_epoch = gr.Number(value=50, visible=False, precision=0) #50
                dmtet = gr.Checkbox(value=True, visible=False)
            finetune_button = gr.Button(value="finetune")
            file_output = gr.File(visible=False)
            # events
            image_slider_input.change(fn=gradio_fn.update_image_slider, inputs=[max_epoch, workspace_input, image_slider_input, dmtet], outputs=[image_output])
            workspace_input.focus(fn=gradio_fn.update_workspaces_list, outputs=[workspace_input])
            workspace_input.change(fn=gradio_fn.update_max_epoch_fetch, inputs=[workspace_input], outputs=[max_epoch])
            workspace_input.change(fn=gradio_fn.update_iters_fetch, inputs=[workspace_input], outputs=[iters_input])
            finetune_button.click(fn=gradio_fn.finetune_3d_model, 
                                  inputs=[workspace_input, seed_input, tet_grid_size_input, iters_input, lr_input, backbone_input, optim_input, fp16_input, max_epoch], 
                                  outputs=[image_slider_input, file_output]) # the output is to update the maximum value of the image_slider which will trigger the "update_image_slider" event to load the image
            
        with gr.Tab("dmtet finetuning 3d model (multi)"):
            with gr.Row():
                image_output = gr.Image(height=512, width=512)
            image_slider_input = gr.Slider(label="image (slide to change angle)", minimum=1, maximum=1, step=1)
            with gr.Row():
                workspace_input = gr.Dropdown(label="workspace name", choices=workspaces)
                seed_input = gr.Number(label="seed", precision=0, value=None)
            tet_grid_size_input = gr.Dropdown(label="tet_grid_size", choices=["32", "64", "128", "256"], value="128")
            with gr.Row():
                iters_input = gr.Number(label="iters (iterations)", value=10000, precision=0, step=1) #5000
                lr_input = gr.Number(label="lr (learning rate)", value=1e-3) #1e-3
            with gr.Accordion(label="advanced", open=False):
                backbone_input = gr.Dropdown(label="backbone (NeRFNetwork)", choices=["grid", "vanilla", "grid_tcnn", "grid_taichi"], value="grid") #nVidia
                optim_input = gr.Dropdown(label="optim (optimizer)", choices=["adan", "adam"], value="adan")
                fp16_input = gr.Checkbox(label="fp16 (use float16 for training)", value=True)
                max_epoch = gr.Number(value=50, visible=False, precision=0) #50
                dmtet = gr.Checkbox(value=True, visible=False)
            finetune_button = gr.Button(value="finetune")
            file_output = gr.File(visible=False)
            # events
            image_slider_input.change(fn=gradio_fn.update_image_slider, inputs=[max_epoch, workspace_input, image_slider_input, dmtet], outputs=[image_output])
            workspace_input.focus(fn=gradio_fn.update_workspaces_list, outputs=[workspace_input])
            workspace_input.change(fn=gradio_fn.update_max_epoch_fetch, inputs=[workspace_input], outputs=[max_epoch])
            workspace_input.change(fn=gradio_fn.update_iters_fetch, inputs=[workspace_input], outputs=[iters_input])
            finetune_button.click(fn=gradio_fn.finetune_3d_model_multi, 
                                  inputs=[workspace_input, seed_input, tet_grid_size_input, iters_input, lr_input, backbone_input, optim_input, fp16_input, max_epoch], 
                                  outputs=[image_slider_input, file_output]) # the output is to update the maximum value of the image_slider which will trigger the "update_image_slider" event to load the image

        with gr.Tab("rod's workflow"):
        #     with gr.Row():
        #         dataset_size_train_input = gr.Number(label="dataset_size_train", value=100, precision=0, step=1) #100
        #         dataset_size_valid_input = gr.Number(label="dataset_size_valid", value=8, precision=0, step=1) #8
        #         dataset_size_test_input = gr.Number(label="dataset_size_test", value=100, precision=0, step=1) #100
        #     with gr.Accordion(label="advanced", open=False):
        #         backbone_input = gr.Dropdown(label="backbone (NeRFNetwork)", choices=["grid", "vanilla", "grid_tcnn", "grid_taichi"], value="grid") #nVidia
        #         optim_input = gr.Dropdown(label="optim (optimizer)", choices=["adan", "adam"], value="adan")
        #         fp16_input = gr.Checkbox(label="fp16 (use float16 for training)", value=True)
        #         max_epoch = gr.Number(value=50, visible=False, precision=0) #50
        #         dmtet = gr.Checkbox(value=False, visible=False)
            
            
            image_input = gr.Image(height=512, width=512)
            with gr.Row():
                image_output_1 = gr.Image(height=512, width=512, visible=False)
                image_output_2 = gr.Image(height=512, width=512, visible=False)
                image_output_3 = gr.Image(height=512, width=512, visible=False)
            with gr.Row():
                image_output_4 = gr.Image(height=512, width=512, visible=False)
                image_output_5 = gr.Image(height=512, width=512, visible=False)
                image_output_6 = gr.Image(height=512, width=512, visible=False)
            radius_input = gr.Number(label="radius", value=0.0) # 3.2
            generate_novel_views_button = gr.Button(value="generate novel views")
            model_output = gr.Image(height=512, width=512, visible=False)
            image_slider_input = gr.Slider(label="image (slide to change angle)", minimum=1, maximum=1, step=1, visible=False)
            with gr.Row():
                workspace_input = gr.Textbox(label="workspace name", value="test", visible=False) #trial_image
                seed_input = gr.Number(label="seed", precision=0, visible=False)
            size_input = gr.Number(label="size (n^2, 64 really recommended.)", value=64, precision=0, step=1, visible=False) #64
            with gr.Row():
                iters_input = gr.Number(label="iters (iterations)", value=5000, precision=0, step=1, visible=False) #5000
                lr_input = gr.Number(label="lr (learning rate)", value=1e-3, visible=False) #1e-3
                batch_size_input = gr.Number(label="batch_size", value=1, precision=0, step=1, visible=False) #1
            with gr.Row():
                dataset_size_train_input = gr.Number(label="dataset_size_train", value=100, precision=0, step=1, visible=False) #100
                dataset_size_valid_input = gr.Number(label="dataset_size_valid", value=8, precision=0, step=1, visible=False) #8
                dataset_size_test_input = gr.Number(label="dataset_size_test", value=100, precision=0, step=1, visible=False) #100
            with gr.Accordion(label="advanced", open=False):
                backbone_input = gr.Dropdown(label="backbone (NeRFNetwork)", choices=["grid", "vanilla", "grid_tcnn", "grid_taichi"], value="grid", visible=False) #nVidia
                optim_input = gr.Dropdown(label="optim (optimizer)", choices=["adan", "adam"], value="adan", visible=False)
                fp16_input = gr.Checkbox(label="fp16 (use float16 for training)", value=True, visible=False)
                max_epoch = gr.Number(value=50, visible=False, precision=0) #50
                dmtet = gr.Checkbox(value=False, visible=False)
            generate_3d_model_button = gr.Button(value="generate 3d model", visible=False)
            # events
            image_slider_input.change(fn=gradio_fn.update_image_slider, inputs=[max_epoch, workspace_input, image_slider_input, dmtet], outputs=[image_output])
            workspace_input.focus(fn=gradio_fn.update_workspaces_list, outputs=[workspace_input])
            workspace_input.change(fn=gradio_fn.update_max_epoch_fetch, inputs=[workspace_input], outputs=[max_epoch])
            workspace_input.change(fn=gradio_fn.update_iters_fetch, inputs=[workspace_input], outputs=[iters_input])
            generate_novel_views_button.click(fn=gradio_fn.generate_novel_view_multi, 
                                              inputs=[image_input, radius_input], 
                                              outputs=[workspace_input, seed_input, size_input, iters_input, lr_input, batch_size_input, dataset_size_train_input, 
                                                       dataset_size_valid_input, dataset_size_test_input, backbone_input, optim_input, fp16_input, 
                                                       generate_3d_model_button, image_output_1, image_output_2, image_output_3, image_output_4, image_output_5, image_output_6])
            generate_3d_model_button.click(fn=gradio_fn.generate_3d_model_multi,
                                           inputs=[image_output_1, image_output_2, image_output_3, image_output_4, image_output_5, image_output_6, radius_input,
                                                   workspace_input, seed_input, size_input, iters_input, lr_input, batch_size_input, dataset_size_train_input, 
                                                   dataset_size_valid_input, dataset_size_test_input, backbone_input, optim_input, fp16_input, max_epoch],
                                           outputs=[model_output, image_slider_input, file_output])
    
    demo.launch()
    
    # rmdir temp folder
    delete_directory("temp/")
    # clear gpu vram
    torch.cuda.empty_cache()