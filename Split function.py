import splitfolders
input_folder = 'Split'

splitfolders.ratio(input_folder, output="Splitted", seed=1337, ratio=(.6, .2, .2), group_prefix=None)