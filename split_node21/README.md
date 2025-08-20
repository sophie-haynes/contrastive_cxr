# Split Node21

Due to memory limitations on github, it was not possible to upload the dataset directly here. 

Experiment notebooks expect the following structure:
- arch_seg
    - /train
        - /nodule
            - /nXXXX
                - /lung_l.png
                - /lung_r.png
        - /normal
            - /cXXXX
                - /lung_l.png
                - /lung_r.png
    - /test
        - /nodule
            - /nXXXX
                - /lung_l.png
                - /lung_r.png
        - /normal
            - /cXXXX
                - /lung_l.png
                - /lung_r.png
    -/process_log.csv
- crop ...
- lung_seg ... 
