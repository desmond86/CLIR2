# Authors:
# Hai Dong Luong (573780) <hai-ld>
# Desmond Putra () <dputra>
# Andrew Vadnal (326558) <avadnal>

class ReorderingModel:
    def __init__(self):
        pass

    def score(self, prev_phrase, next_phrase):
        """Calculate reordering cost based on distance between phrases."""
        dist = next_phrase[0][0] - prev_phrase[1] - 1
        return -(abs(dist))
