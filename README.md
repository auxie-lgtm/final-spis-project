# final-spis-project

This project is a singing evaluation model. 

Description: 
The model accepts an audio file alongside a user-inputted message to allow an LLM to evaluate the singing through conversion into a spectrogram and transcript. 

TO-DO:
- allow uploading of audio file in lieu of requiring pathname
- create website to facilitate uploading and allow further user interaction

Current Files:
- ai.py - serves as the "prompt manager"; focuses on text generation and artificial intelligence usage
- app.py - serves as the backend manager behind the web application
- audio.py - serves as the "audio manager"; processes audio file and gives spectrograph and transcript for LLM in ai.py to examine
- augment_dataset.py - expands the dataset that the model trains on, allowing for more concise and less overfitted data
- class_identifier_2.py - a helper file intended to provide a baseline for karaoke_identifier.py to expand upon, hence the inheritance
- karaoke_classifer.py - serves as the classifier that categorizes the performance into different "classes" based on how good the performance is; contains a small snippet of code to run to train the model (only once!)
- main.py - the file to run

.
- memories_of_kindness.mp3 - a Japanese audio file; serves as a test
- sight-singing-vocal-data directory - a directory with one (or more) dataset(s); serves to give training/validation/test cases to future ML

.
- singer_grade_model.keras - saved model

.
- static directory - for css files
    - music-sheet.jpg - gives background
    - web_deco.css - gives style of website

.
- templates - provides html files
    - ddr-rank-reveal.html - (FILLER) gives animation of classification score
    - eval.html - (FILLER) gave a page to evaluate performances
    - index.html - the home page
    - layout.html - (FILLER) gave a consistent layout for multiple html files
    - result.html - the result page displaying score and LLM response
