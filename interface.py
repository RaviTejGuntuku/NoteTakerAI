import transcribe_audio as ta

file = input("Enter file name: ")

t1 = ta.Transcribe(file, True)

print("Output saved to " + t1.transcription_name)
