wavs = [AudioSegment.from_wav(wav['path']) for wav in wavset]
combined = wavs[0]

for wav in wavs[1:]:
    combined = combined.append(wav) 