import whisper


class Transcribe:
    text = ""

    def __init__(self, audio_file, should_auto_execute_transcription_and_save=False):
        self.audio_file = audio_file
        self.transcription_name = audio_file.split(
            ".", 1)[0] + "_transcription.txt"

        if (should_auto_execute_transcription_and_save):
            self.transcribe()
            self.save_output_txt()

    def transcribe(self):
        model = whisper.load_model("base")

        result = model.transcribe(self.audio_file, fp16=False)
        self.text = result["text"]
        return self.text

    def save_output_txt(self):
        output_file = open(self.transcription_name, "w")

        string_to_transcribe = ""

        for i in self.text:
            string_to_transcribe += i
            # if (len(string_to_transcribe) >= 100):
            #     if (i == " "):
            #         output_file.write(string_to_transcribe + "\n")
            #         string_to_transcribe = ""

        output_file.write(string_to_transcribe + "\n")
        output_file.close()
