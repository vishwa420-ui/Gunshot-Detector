import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Configure voice properties
engine.setProperty('rate', 150)  # Speed (default ~200)
engine.setProperty('volume', 10.0)  # Max volume (0.0 to 1.0)

# Speak text
text = "Gunshot detected. Please take immediate action."
engine.say(text)

# Wait until speaking is finished
engine.runAndWait()