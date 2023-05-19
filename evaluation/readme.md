### Improvement:

- Usage

  - r_precision.py <br>
  For prompt seperation <br>
  --text is for the prompt following the author of stable dream fusion <br>
  --workspace is the workspace folder which will be created for every prompt fed into stable dreamfusion <br>
  --latest is which ckpt is used. Stable dream fusion record every epoch data. Normally is ep0100 unless the training is not finished or we further extend the training <br>
  --mode has choices of rgb and depth which is correspondent to color and texture result as original paper Figure 5: Qualitative comparison with baselines. <br>
  --clip has choices of clip-ViT-B-32, CLIP B/16, CLIP L/14, same as original paper <br>

      ```bash
      python Prompt.py --text "matte painting of a castle made of cheesecake surrounded by a moat made of ice cream" --workspace ../castle --latest ep0100 --mode rgb --clip clip-ViT-B-32
      ```

  - Prompt.py (model name case sensitive) <br>
  For prompt seperation <br> <br>
  --text is for the prompt following the author of stable dream fusion <br>
  --model is for choose the pretrain models <br>

      ```bash
      python Prompt.py --text "a dog is in front of a rabbit" --model vlt5
      python Prompt.py --text "a dog is in front of a rabbit" --model bert
      python Prompt.py --text "a dog is in front of a rabbit" --model XLNet
      ```


  - mesh_to_video.py <br>
  --center_obj IS THE CENTER OBJECT <br>
  --surround_obj IS THE SURROUNDING OBJECT SUBJECT TO CHANGE <br>
  --transform_vector THE X Y Z 3d vector for transform <br>

      ```bash
      python mesh_to_video.py --center_obj 'mesh_whiterabbit/mesh.obj' --surround_obj 'mesh_snake/mesh.obj' --transform_vector [1,0,0]
      ```
