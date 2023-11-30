# Scene Understanding 

## CLIP

OpenAI introduced a model called [Contrastive Language-Image Pre-training](https://openai.com/research/clip) that is extensively described in their [paper Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf). 

Its implementation [is supported by HF]https://huggingface.co/docs/transformers/model_doc/clip) and the original code is [here](https://github.com/openai/CLIP). You dont need to use the original code but you can if you want to.

This model learns the relationship between a whole sentence and the image it describes; Given an input sentence it will be able to retrieve the most related images corresponding to that sentence. The system is trained on full sentences instead of single classes like car, dog, etc. The intuition is that when trained on whole sentences, the model can learn a lot more things and finds some pattern between images and texts.

The following video is a very good exaplanation as to how CLIP compares with permuations on the image and text embeddings and its performance characteristics.

```{eval-rst}

.. youtube:: https://www.youtube.com/watch?v=BcfAkQagEWU
   :width: 400
   :height: 300
   :align: center
```


# Understanding Video Scenes using VideoCLIP

 
[VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding](https://arxiv.org/abs/2109.14084)



