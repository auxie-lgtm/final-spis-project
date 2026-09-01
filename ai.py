'''
LLM evaluation layer.

This file is responsible for turning the raw audio analysis into a natural-language explanation.
The model does not directly judge the singer from the waveform itself; instead, it receives a
structured summary of statistics, approximate scores, and transcript text and turns that into
coaching feedback.
'''

# Import all libraries for the LLM evaluation flow.
# transformers: provides the text-generation pipeline for TinyLlama.
# torch: used for device handling and model execution.
# audio: provides the audio-processing pipeline and metrics.
# KaraokeClassifier: provides the trained-model and heuristic-score logic.

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

    LETTER_SCORES = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

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
        self.__overall_grade = None  # Initialize the overall grade attribute

    @staticmethod
    def _normalize_grade(grade):
        if grade is None:
            return "F"
        normalized = str(grade).strip().upper()
        if normalized in {"S", "A", "B", "C", "D", "F"}:
            return normalized
        if normalized == "D/F":
            return "D"
        if normalized.startswith("D"):
            return "D"
        if normalized.startswith("F"):
            return "F"
        if normalized.startswith("C"):
            return "C"
        if normalized.startswith("B"):
            return "B"
        if normalized.startswith("A"):
            return "A"
        if normalized.startswith("S"):
            return "S"
        return "F"

    # Converts the rankings to a score
    @staticmethod
    def _letter_to_score(letter):
        normalized = PromptManager._normalize_grade(letter)
        return PromptManager.LETTER_SCORES.get(normalized, 0)

    # Converts the scores to a ranking
    @staticmethod
    def _score_to_letter(score):
        if score >= 4.5:
            return "S"
        if score >= 3.5:
            return "A"
        if score >= 2.5:
            return "B"
        if score >= 1.5:
            return "C"
        if score >= 0.5:
            return "D"
        return "F"

    # Function that builds the template response that the LLM gives out at the end
    def _build_template_response(self, overall_grade, dataset_grade, standalone_grade, metrics, blended_score=None, audio_analysis=None, message=None):
        pitch = float(metrics.get("pitch_score", 0.0))
        beat = float(metrics.get("beat_score", 0.0))
        clarity = float(metrics.get("clarity_score", 0.0))
        consistency = float(metrics.get("consistency_score", 0.0))

        metric_scores = {
            "pitch": pitch,
            "beat": beat,
            "clarity": clarity,
            "consistency": consistency,
        }

        strongest_metric, strongest_value = max(metric_scores.items(), key=lambda item: item[1])
        weakest_order = sorted(metric_scores.items(), key=lambda item: item[1])

        if strongest_metric == "pitch":
            strength_text = "strong pitch control and stable vocal placement."
        elif strongest_metric == "beat":
            strength_text = "steady timing and strong rhythmic control."
        elif strongest_metric == "clarity":
            strength_text = "clear diction and confident tone."
        else:
            strength_text = "good consistency across the phrase."

        improvement_text = {
            "pitch": "focus on pitch stability during sustained notes.",
            "beat": "tighten timing alignment with the beat.",
            "clarity": "improve diction and tonal clarity at phrase endings.",
            "consistency": "smooth out phrase-to-phrase consistency.",
        }

        # Pick the weakest metrics first, then fill any remaining spaces with the next most relevant coaching points.
        ordered_improvements = [
            improvement_text.get(name, "keep working on vocal control.")
            for name, _ in weakest_order
        ]
        if len(ordered_improvements) < 3:
            ordered_improvements.extend([
                "keep the phrase steady and relaxed.",
                "maintain your current tone and support.",
            ])
        ordered_improvements = ordered_improvements[:3]

        if consistency >= 75:
            reason_text = (
                f"The model classification was {dataset_grade}, the standalone classification was {standalone_grade}, "
                f"and the final verdict settled on {overall_grade} because the performance showed strong consistency and control "
                "across the main vocal phrases."
            )
        elif strongest_value >= 75:
            reason_text = (
                f"The model classification was {dataset_grade}, the standalone classification was {standalone_grade}, "
                f"and the final verdict settled on {overall_grade} because the main strengths were solid, but a few weak spots "
                "still limited the overall result."
            )
        else:
            reason_text = (
                f"The model classification was {dataset_grade}, the standalone classification was {standalone_grade}, "
                f"and the final verdict settled on {overall_grade} because the performance was uneven across pitch, timing, and "
                "vocal control, so the combined evidence favored a lower grade."
            )

        if blended_score is not None:
            reason_text += f" Combined score: {blended_score:.2f}."

        if audio_analysis is not None:
            transcript = str(audio_analysis.get("transcript", "")).strip()
            pitch_stats = audio_analysis.get("pitch_statistics", {}) or {}
            if transcript:
                transcript_snippet = transcript[:60]
                reason_text += f" Transcript note: {transcript_snippet}."
            if pitch_stats:
                mean_pitch = pitch_stats.get("average_hz")
                if mean_pitch is not None:
                    reason_text += f" Mean pitch: {mean_pitch} Hz."

        if message:
            reason_text += f" User note: {message.strip()[:80]}."

        return (
            f"Model classification: {dataset_grade}\n"
            f"Standalone classification: {standalone_grade}\n"
            f"Overall verdict: {overall_grade}\n"
            f"Strengths: {strength_text}\n"
            "Improvements:\n"
            f"1. {ordered_improvements[0]}\n"
            f"2. {ordered_improvements[1]}\n"
            f"3. {ordered_improvements[2]}\n"
            f"Why this grade: {reason_text}\n"
            "Note: approximate because there is no reference melody."
        )

    def _combine_grades(self, dataset_grade, dataset_confidence, standalone_grade, standalone_confidence, audio_analysis=None):
        dataset_score = self._letter_to_score(dataset_grade)
        standalone_score = self._letter_to_score(standalone_grade)

        # Give a little more weight to the standalone heuristic for new data, because the
        # dataset-trained model is only reliable within the project's training distribution.
        dataset_weight = 0.45 if dataset_confidence > 0.5 else 0.2
        standalone_weight = 1.0 - dataset_weight

        blended_score = (dataset_score * dataset_weight) + (standalone_score * standalone_weight)

        # The processed audio analysis provides a real signal about sung stability and voiced coverage.
        # Use it as a small adjustment so the final grade reflects actual vocal behavior, not just the
        # dataset heuristics. This is still conservative: we do not treat it as exact pitch truth.
        if audio_analysis is not None:
            pitch_stats = audio_analysis.get("pitch_statistics", {}) or {}
            transcript = str(audio_analysis.get("transcript", "")).strip()
            voiced_percentage = float(pitch_stats.get("voiced_percentage", 0.0) or 0.0)
            variation_hz = float(pitch_stats.get("variation_hz", 0.0) or 0.0)
            has_transcript = 1.0 if transcript else 0.0

            audio_adjustment = 0.0
            if voiced_percentage >= 70:
                audio_adjustment += 0.25
            elif voiced_percentage < 35:
                audio_adjustment -= 0.35

            if variation_hz > 90:
                audio_adjustment -= 0.35
            elif variation_hz < 35:
                audio_adjustment += 0.15

            if has_transcript:
                audio_adjustment += 0.10

            blended_score += audio_adjustment

        blended_score = max(0.0, min(5.0, blended_score))
        overall_grade = self._score_to_letter(blended_score)
        self.__overall_grade = overall_grade  # Store the overall grade for external access
        return overall_grade, blended_score

    # Evaluate one uploaded song.
    # This method collects the transcript, objective metrics, and score approximations,
    # builds a structured prompt, and sends it to the LLM for a final interpretation.
    def evaluate_song(self, audio_path, message=None):
        try:
            audio_analysis = self.audio_processor.process_audio(audio_path)
            metrics = self.audio_processor.evaluate_audio_metrics(audio_path)

            dataset_grade, dataset_confidence = self.classifier.estimate_standalone_grade(audio_path)
            if self.classifier.dataset_model is not None:
                try:
                    dataset_grade, dataset_confidence = self.classifier.predict_grade(audio_path)
                except Exception:
                    dataset_grade, dataset_confidence = self.classifier.estimate_standalone_grade(audio_path)

            dataset_grade = self._normalize_grade(dataset_grade)

            standalone_grade, standalone_confidence = self.classifier.estimate_standalone_grade(audio_path)
            standalone_grade = self._normalize_grade(standalone_grade)

            overall_grade, blended_score = self._combine_grades(
                dataset_grade,
                dataset_confidence,
                standalone_grade,
                standalone_confidence,
                audio_analysis,
            )
        except Exception as exc:
            return f"Error occurred while processing audio: {exc}"

        # The model still exists for future richer generation, but the final response is kept
        # deterministic and driven by the actual metrics so it stays accurate and brief.
        response = self._build_template_response(
            overall_grade,
            dataset_grade,
            standalone_grade,
            metrics,
            blended_score,
            audio_analysis,
            message,
        )
        return response

    def get_overall_grade(self):
        return self.__overall_grade


    # CLI-style entry point used when the script is run directly.
    # It asks for a user description and a file path, then invokes the full evaluation pipeline.
    def prompt(self):
        message = input("Describe your karaoke performance: ")
        audio_path = input("Path to the audio file: ")
        return self.evaluate_song(audio_path, message)
