# final-spis-project

This project is a singing evaluation model. 

Description: 
The model accepts an audio file alongside a user-inputted message to allow an LLM to evaluate the singing through conversion into a spectrogram and transcript. 

TO-DO:
- integrate ML
  - integrate the LLM in karaoke_classifier.py into the prompt manager used in ai.py
- allow uploading of audio file in lieu of requiring pathname
- create website to facilitate uploading and allow further user interaction

Current Files:
- ai.py - serves as the "prompt manager"; focuses on text generation and artificial intelligence usage
- audio.py - serves as the "audio manager"; processes audio file and gives spectrograph and transcript for LLM in ai.py to examine
- karaoke_classifer.py - serves as the classifier that categorizes the performance into different "classes" based on how good the performance is
- main.py - the file to run
- memories_of_kindness.mp3 - a Japanese audio file; serves as a test
- sight-singing-vocal-data directory - a directory with one (or more) dataset(s); serves to give training/validation/test cases to future ML
