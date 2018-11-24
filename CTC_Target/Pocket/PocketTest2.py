from __future__ import print_function
import os
from pocketsphinx import Pocketsphinx, get_model_path, get_data_path

model_path = get_model_path()
data_path = get_data_path()

config = {
    'hmm': os.path.join(model_path, 'en-us'),
    'lm': r'C:\Users\BZT\Desktop\speech_segment\5446.lm',
    'dict': r'C:\Users\BZT\Desktop\speech_segment\5446.dic'
}

ps = Pocketsphinx(**config)
ps.decode(
    audio_file=r'C:\Users\BZT\Desktop\speech_segment\speech_segment\Ses01F_impro01_M013.wav',
    buffer_size=2048,
    no_search=False,
    full_utt=False
)

# print(ps.segments())  # => ['<s>', '<sil>', 'go', 'forward', 'ten', 'meters', '</s>']
# print('Detailed segments:', *ps.segments(detailed=True), sep='\n')
for sample in ps.segments(detailed=True):
    for subsample in sample:
        print(subsample, end='\t')
    print()

print(ps.hypothesis())  # => go forward ten meters
# print(ps.probability())  # => -32079
# print(ps.score())  # => -7066
# print(ps.confidence())  # => 0.04042641466841839

# print(*ps.best(count=10), sep='\n')
