import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:

    # Implement the Load function here for the model
    def initialize(self):
        self.generator = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1",device=0)
        print("This is Initialize Function", flush=True)

    
    # Function to perform inference 
    def infer(self, inputs):
        prompt = inputs["prompt"]
        pipeline_output = self.generator(prompt, do_sample=True, min_length=20)
        generated_txt = pipeline_output[0]["generated_text"]

        return {"generated_text": generated_txt}

    # perform any cleanup activity here
    def finalize(self,args):
        self.pipe = None
