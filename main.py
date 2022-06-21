import torchaudio
from speechbrain.pretrained.interfaces import foreign_class
import warnings

warnings.filterwarnings("ignore")

classifier = foreign_class(
    source=".\\",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
)
wav_file = "sld.wav"
metadata = torchaudio.info(wav_file)
results = classifier.classify_file(wav_file, metadata.sample_rate, metadata.num_frames)
for r in results:
    print(r[0], r[1], r[3])
