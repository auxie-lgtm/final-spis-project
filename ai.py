import re

# importing transformers and torch libraries
# transformers library provides the llm
# torch gives device management and hopefully provides machine learning capabilities

from transformers import pipeline
import torch
import audio

class PromptManager:
    '''
    The PromptManager class provides the possible prompts of the artificial intelligence
    that evaluates karaoke performance based on user input. The user inputs a message
    alongside video or audio files and then the AI uses this message to evaluate the karaoke performance.
    '''

    # The model used for text generationis TinyLlama. 

    __MODEL = "TinyLlama/tinyllama-1.1B-Chat-v1.0"

    # Device management done with torch. 
    # Tries to use bfloat16 if CUDA is available, otherwise uses float32.

    pipe_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Generates the model

    def __init__(self):
        self.generate = pipeline(
            task="text-generation",
            model=self.__MODEL,
            dtype=self.pipe_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        self.audio_processor = audio.AudioProcessor()

    # The input message. The user is given an input prompt to summarize their day;
    # the message provided is then used to generate a response from the model. 

    def prompt(self):
        # change into a prompt that allows user to input a message
        # alongside an audio file to evaluate their karaoke performance.
        
        message = input("Describe your karaoke performance: ")
        try:
            audio_path = input("Path to the audio file: ")
            audio_analysis = self.audio_processor.process_audio(audio_path)
        except Exception as e:
            print(f"Error occurred while processing audio: {e}")
            return "An error occurred while processing the audio file."

        user_content = (
            f"User message: {message}\n\n"
            f"Audio analysis: {audio_analysis}\n\n"
            "Evaluate the performance using the available information."
        )

        result = self.generate(
            [{"role": "system", "content": "You are an expert karaoke evaluator. Assess the user's performance using their message, transcript, and extracted audio measurements. Do not claim that these measurements prove pitch accuracy without a reference melody."}, {"role": "user", "content": user_content}],
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )

        # The response is transformed into a string and printed. 

        answer = result[0]["generated_text"]
        if isinstance(answer, list):
            answer = answer[-1].get("content", "")
        elif isinstance(answer, str):
            answer = answer
        else:
            answer = str(answer)

        # intended to make a cleaner answer
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        cleaned_sentences = []
        for sentence in sentences:
            if not cleaned_sentences or sentence != cleaned_sentences[-1]:
                cleaned_sentences.append(sentence)
        return " ".join(cleaned_sentences)
