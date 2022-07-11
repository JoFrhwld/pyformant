from pyformant import formant

vowelfile = "vowel.wav"

test = formant.VowelLike(path=vowelfile, max_formant=5500)
df = test.formant_df