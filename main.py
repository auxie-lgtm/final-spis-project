'''
Karaoke AI

This code provides the possible prompts of the artificial intelligence
that evaluates karaoke performance based on user input. The user inputs a message
alongside video or audio files and then the AI uses this message to evaluate the karaoke performance. 
'''

import ai

p = ai.PromptManager()
print(p.prompt())