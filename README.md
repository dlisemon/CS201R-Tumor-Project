# CS201R-Tumor-Project
Central repository for all project files. 


## Ideas:
- I think this could be a good place to keep track of goals/tasks, as well as what each of us are working on
- Once we have all looked at the data, I think the first step would be to start deciding on features and firguring out what kind of data preprocessing is required
- Then, I think we could start looking what models we want to use (maybe code up a few scripts to run some debugging/testing and comparing accuracy?)
- Maybe we could add some deadlines for ourselves to stay on-track? Break things down into smaller tasks? 


## Deadlines:
- Group Project Progress Report: 11/7/23
- Hardcopy and Presentation: 12/12/23 (my wedding anniversary, lol)

## Questions to Answer:
1) Targets = Glioma grade (1, 2, 3, or 4)
2) Decide on a final feature list. Working list:
  - 'Tumor location' // real (1, 2, 3, 4, 5 = frontal, temporal, parietal, occipital, deep)
  - 'Tumor area' // continuous (in cm^2)
  - 'Tumor sphericity // continuous (scale from 0->1 with 1 representing a perfect sphere)
  - 'Tumor texture' // continuous
  - 'Tumor intensity mean' // continuous
  -  'GLCM Contrast' // continuous
  -  'Class' // real (1, 2, 3, 4, denoting glioma grade)
    More ideas:
      - Peritumoral edema: Presence and extent of edema is associated with higher grade. Segmenting and measuring edema volume could be a useful feature.          (we could stick to a binary representation for simplicity, 1 = edema present, 0 = edema absent
      - Diffusion properties: Diffusion MRI measures microstructural characteristics and can distinguish low vs high grade. Adding diffusion properties            like ADC values as features could help.
      - Perfusion characteristics: Higher perfusion and neoangiogenesis occurs with higher grade tumors. Perfusion imaging features like CBV would be
        useful.

