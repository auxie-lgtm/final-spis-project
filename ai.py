'''
LLM evaluation layer.

This file is responsible for turning the raw audio analysis into a natural-language explanation.
The model does not directly judge the singer from the waveform itself; instead, it receives a
structured summary of statistics, approximate scores, and transcript text and turns that into
coaching feedback.
'''

# Import all libraries for the LLM evaluation flow.
# re: used to clean the generated text into readable sentences.
# transformers: provides the text-generation pipeline for TinyLlama.
# torch: used for device handling and model execution.
# audio: provides the audio-processing pipeline and metrics.
# KaraokeClassifier: provides the trained-model and heuristic-score logic.

import re
from transformers import pipeline
import torch
import audio
from karaoke_classifier import KaraokeClassifier

class PromptManager:
    '''
    PromptManager creates the final evaluation prompt sent to the LLM.

    It gathers:
    - transcript text from Whisper
    - pitch and tempo statistics from librosa
    - rough score estimates from the dataset model or heuristic fallback
    - user description of the performance

    The LLM then converts all of that into a short coaching summary.
    '''

    # TinyLlama is the language model used to generate the final text explanation.
    __MODEL = "TinyLlama/tinyllama-1.1B-Chat-v1.0"

    # Use bfloat16 on CUDA when possible, otherwise fall back to float32.
    # This is the standard lightweight inference setup for transformer models on supported hardware.
    pipe_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Initialize the LLM pipeline and the audio/classifier helpers.
    def __init__(self):
        self.generate = pipeline(
            task="text-generation",
            model=self.__MODEL,
            dtype=self.pipe_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        self.audio_processor = audio.AudioProcessor()
        self.classifier = KaraokeClassifier()

    # Evaluate one uploaded song.
    # This method collects the transcript, objective metrics, and score approximations,
    # builds a structured prompt, and sends it to the LLM for a final interpretation.
    def evaluate_song(self, audio_path, message):
        try:
            audio_analysis = self.audio_processor.process_audio(audio_path)
            metrics = self.audio_processor.evaluate_audio_metrics(audio_path)

            dataset_grade, dataset_confidence = self.classifier.estimate_standalone_grade(audio_path)
            if self.classifier.dataset_model is not None:
                try:
                    dataset_grade, dataset_confidence = self.classifier.predict_grade(audio_path)
                except Exception:
                    dataset_grade, dataset_confidence = self.classifier.estimate_standalone_grade(audio_path)

            standalone_grade, standalone_confidence = self.classifier.estimate_standalone_grade(audio_path)
        except Exception as exc:
            return f"Error occurred while processing audio: {exc}"

        # Build a prompt that makes the score sources explicit and tells the model not to
        # overclaim exact pitch accuracy when no reference melody is available.
        prompt = (
            f"User description: {message}\n\n"
            f"Approximate dataset-model grade: {dataset_grade}\n"
            f"Dataset confidence: {dataset_confidence:.2f}\n\n"
            f"Approximate standalone heuristic grade: {standalone_grade}\n"
            f"Standalone confidence: {standalone_confidence:.2f}\n\n"
            f"Audio metrics:\n"
            f"- pitch score: {metrics['pitch_score']}\n"
            f"- beat score: {metrics['beat_score']}\n"
            f"- clarity score: {metrics['clarity_score']}\n"
            f"- consistency score: {metrics['consistency_score']}\n\n"
            f"Transcript:\n{audio_analysis['transcript']}\n\n"
            f"Pitch statistics:\n{audio_analysis['pitch_statistics']}\n\n"
            "Important: there is no reference melody or exact note target. Treat both scores as rough indicators of tonal stability and vocal control, not precise pitch-accuracy measurements.\n\n"
            "Give a concise but helpful karaoke evaluation. Include:\n"
            "1. an overall verdict\n"
            "2. what the singer did well\n"
            "3. three specific improvements\n"
            "4. a brief explanation of how the audio metrics and rough score sources support the verdict\n"
            "5. a sentence stating that the score is approximate because there is no reference melody."
        )

        # Run the text-generation model with a limited output length and basic repetition control.
        result = self.generate(
            [
                {"role": "system", "content": "You are an expert karaoke coach and music evaluator."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
        )

        # Convert the generated text into a single clean string and remove duplicated sentences.
        answer = result[0]["generated_text"]
        if isinstance(answer, list):
            answer = answer[-1].get("content", "")
        elif not isinstance(answer, str):
            answer = str(answer)

        # intended to make a cleaner answer
        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        cleaned_sentences = []
        for sentence in sentences:
            if not cleaned_sentences or sentence != cleaned_sentences[-1]:
                cleaned_sentences.append(sentence)
        return " ".join(cleaned_sentences)

    # CLI-style entry point used when the script is run directly.
    # It asks for a user description and a file path, then invokes the full evaluation pipeline.
    def prompt(self, message, audio_path): #Retrieved from app.py
        return self.evaluate_song(audio_path, message)
